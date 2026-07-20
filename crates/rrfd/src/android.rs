use jni::ids::JStaticMethodID;
use jni::objects::JClass;
use jni::refs::Global;
use jni::signature::Primitive;
use jni::sys::{jint, jvalue};
use jni::{EnvUnowned, jni_sig, jni_str};
use lazy_static::lazy_static;
use std::io;
use std::os::fd::FromRawFd;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use tokio::fs::File;
use tokio::sync::oneshot;

// Keep activity request codes in a nonzero 16-bit range. On wrap, a late
// callback can only match after every code in the range has been issued; the
// Java bridge rejects out-of-range codes and Rust still verifies the active ID.
const FIRST_REQUEST_ID: jint = 0x1000;
const LAST_REQUEST_ID: jint = 0x7fff;

struct PendingPick {
    id: jint,
    sender: oneshot::Sender<Option<File>>,
}

struct PickerState {
    next_id: jint,
    pending: Option<PendingPick>,
}

impl Default for PickerState {
    fn default() -> Self {
        Self {
            next_id: FIRST_REQUEST_ID,
            pending: None,
        }
    }
}

struct PendingPickGuard {
    id: jint,
}

impl Drop for PendingPickGuard {
    fn drop(&mut self) {
        if let Ok(mut state) = PICKER_STATE.lock()
            && state
                .pending
                .as_ref()
                .is_some_and(|pick| pick.id == self.id)
        {
            state.pending.take();
        }
    }
}

lazy_static! {
    static ref VM: RwLock<Option<Arc<jni::JavaVM>>> = RwLock::new(None);
    static ref PICKER_STATE: Mutex<PickerState> = Mutex::new(PickerState::default());
    static ref START_FILE_PICKER: RwLock<Option<JStaticMethodID>> = RwLock::new(None);
    static ref FILE_PICKER_CLASS: RwLock<Option<Global<JClass<'static>>>> = RwLock::new(None);
}

#[allow(unused)]
pub fn jni_initialize(vm: Arc<jni::JavaVM>) {
    let (class, method) = vm
        .attach_current_thread(|env| -> jni::errors::Result<_> {
            let class = env.find_class(jni_str!("com/splats/app/FilePicker"))?;
            let method =
                env.get_static_method_id(&class, jni_str!("startFilePicker"), jni_sig!("(I)V"))?;
            Ok((env.new_global_ref(class)?, method))
        })
        .expect("Cannot initialize the file picker JNI data");
    *FILE_PICKER_CLASS
        .write()
        .expect("Failed to write JNI data.") = Some(class);
    *START_FILE_PICKER
        .write()
        .expect("Failed to write JNI data.") = Some(method);
    *VM.write().unwrap() = Some(vm);
}

#[allow(unused)]
pub(crate) async fn pick_file() -> std::io::Result<File> {
    let (sender, receiver) = oneshot::channel();
    let request_id = {
        let mut state = PICKER_STATE
            .lock()
            .map_err(|_| io::Error::other("Failed to lock file picker state"))?;
        if state.pending.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "A file picker request is already active",
            ));
        }
        let request_id = state.next_id;
        state.next_id = if request_id == LAST_REQUEST_ID {
            FIRST_REQUEST_ID
        } else {
            request_id + 1
        };
        state.pending = Some(PendingPick {
            id: request_id,
            sender,
        });
        request_id
    };
    // Dropping the future clears this request if Java never calls back.
    let _pending = PendingPickGuard { id: request_id };

    // Call method. Be sure this is scoped so we drop all guards before waiting.
    {
        let java_vm = VM
            .read()
            .unwrap()
            .clone()
            .expect("Failed to initialize Java VM");
        java_vm
            .attach_current_thread(|env| -> jni::errors::Result<()> {
                let class = FILE_PICKER_CLASS
                    .read()
                    .expect("Failed to initialize FilePicker class");
                let method = START_FILE_PICKER
                    .read()
                    .expect("Failed to initialize FilePicker method");

                // SAFETY: The method ID was resolved against this global class
                // reference with the `(I)V` signature during initialization.
                unsafe {
                    env.call_static_method_unchecked(
                        class.as_ref().expect("Failed to get class reference"),
                        method.as_ref().expect("Failed to get method reference"),
                        jni::signature::ReturnType::Primitive(Primitive::Void),
                        &[jvalue { i: request_id }],
                    )?;
                }
                Ok(())
            })
            .map_err(|e| io::Error::other(format!("JNI error: {e:?}")))?;
    }
    match receiver.await {
        Ok(Some(file)) => Ok(file),
        Ok(None) => Err(io::Error::new(io::ErrorKind::NotFound, "No file selected")),
        Err(_) => Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "File picker result channel closed",
        )),
    }
}

#[unsafe(no_mangle)]
extern "system" fn Java_com_splats_app_FilePicker_onFilePickerResult<'local>(
    _env: EnvUnowned<'local>,
    _class: JClass<'local>,
    request_id: jint,
    fd: jint,
) {
    let file = if fd < 0 {
        None
    } else {
        // Convert the raw file descriptor into a Rust File
        // SAFETY: Pray that JNI gets us a valid file. It will be open
        // when passed to us.
        Some(unsafe { tokio::fs::File::from_raw_fd(fd) })
    };

    // Take the sender so duplicate or stale callbacks are harmless. If the
    // waiting future was dropped, `send` returns the file and closes it here.
    let pending = PICKER_STATE.lock().ok().and_then(|mut state| {
        if state
            .pending
            .as_ref()
            .is_some_and(|pending| pending.id == request_id)
        {
            state.pending.take()
        } else {
            None
        }
    });
    if let Some(pending) = pending {
        let _ = pending.sender.send(file);
    }
}
