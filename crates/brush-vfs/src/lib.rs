mod data_source;

use std::{
    collections::HashMap,
    fmt::Debug,
    io::{self, Cursor, Error},
    path::{Path, PathBuf},
    sync::Arc,
};

use async_zip::base::read::stream::ZipFileReader;
use path_clean::PathClean;
use thiserror::Error;
use tokio::{
    io::{AsyncBufRead, AsyncRead, AsyncReadExt, BufReader},
    sync::Mutex,
};
use tokio_util::compat::{FuturesAsyncReadCompatExt, TokioAsyncReadCompatExt};

pub use data_source::{DataSource, DataSourceError};

// WASM doesn't require Send, but native tokio does.
#[cfg(target_family = "wasm")]
pub trait SendNotWasm {}
#[cfg(target_family = "wasm")]
impl<T> SendNotWasm for T {}
#[cfg(not(target_family = "wasm"))]
pub trait SendNotWasm: Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send> SendNotWasm for T {}

pub trait DynRead: AsyncBufRead + SendNotWasm + Unpin {}
impl<T: AsyncBufRead + SendNotWasm + Unpin> DynRead for T {}

type StreamingReader = Arc<Mutex<Option<Box<dyn DynRead>>>>;

/// Wrapper so `Cursor` can use `Arc<Vec<u8>>` without cloning.
struct ArcVec(Arc<Vec<u8>>);
impl AsRef<[u8]> for ArcVec {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Normalized path key for case-insensitive lookups.
#[derive(Debug, Eq, PartialEq, Hash)]
struct PathKey(String);

impl PathKey {
    fn from_str(path: &str) -> Self {
        let key = path.to_lowercase().replace('\\', "/");
        Self(if key.starts_with('/') {
            key
        } else {
            format!("/{key}")
        })
    }

    fn from_path(path: &Path) -> Self {
        // Lossily convert rather than panicking on non-UTF-8 filenames; the key
        // is only used for case-insensitive lookups.
        Self::from_str(&path.clean().to_string_lossy())
    }
}

async fn read_at_most<R: AsyncRead + Unpin>(reader: &mut R, limit: usize) -> io::Result<Vec<u8>> {
    let mut buffer = vec![0; limit];
    let mut total_read = 0;
    while total_read < limit {
        let bytes_read = reader.read(&mut buffer[total_read..]).await?;
        if bytes_read == 0 {
            break;
        }
        total_read += bytes_read;
    }
    buffer.truncate(total_read);
    Ok(buffer)
}

/// Read from `reader` in chunks of `chunk_size`, calling `parse` on everything
/// read so far after each chunk. Returns the first `Some` value `parse` yields,
/// without consuming the rest of the reader; returns `None` if the reader hits
/// EOF before `parse` succeeds.
pub async fn read_until_parsed<R, T>(
    reader: &mut R,
    chunk_size: usize,
    mut parse: impl FnMut(&[u8]) -> Option<T>,
) -> io::Result<Option<T>>
where
    R: AsyncRead + Unpin,
{
    let mut buf = vec![];
    let mut chunk = vec![0u8; chunk_size];
    loop {
        let n = reader.read(&mut chunk).await?;
        buf.extend_from_slice(&chunk[..n]);
        if let Some(value) = parse(&buf) {
            return Ok(Some(value));
        }
        if n == 0 {
            return Ok(None);
        }
    }
}

enum VfsContainer {
    /// Raw data stored in memory (from zip files)
    InMemory {
        entries: HashMap<PathBuf, Arc<Vec<u8>>>,
    },
    /// A single file being streamed. The reader can only be consumed once.
    Streaming { reader: StreamingReader },
    /// Native directory - reads from disk on demand
    #[cfg(not(target_family = "wasm"))]
    Directory { base_path: PathBuf },
    /// WASM directory - uses File System Access API to read files on demand
    #[cfg(target_family = "wasm")]
    Directory {
        dir_handle: rrfd::wasm::DirectoryHandle,
    },
}

impl Debug for VfsContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InMemory { .. } => f.debug_struct("InMemory").finish(),
            Self::Streaming { .. } => f.debug_struct("Streaming").finish(),
            Self::Directory { .. } => f.debug_struct("Directory").finish(),
        }
    }
}

#[derive(Debug)]
pub struct BrushVfs {
    lookup: HashMap<PathKey, PathBuf>,
    container: VfsContainer,
}

fn is_lookup_path(path: &Path) -> bool {
    path.extension().is_some()
        && !path
            .components()
            .any(|component| component.as_os_str() == "__MACOSX")
}

fn lookup_from_paths(paths: &[PathBuf]) -> HashMap<PathKey, PathBuf> {
    paths
        .iter()
        // Skip directories and __MACOSX metadata
        .filter(|path| is_lookup_path(path))
        // Normalize only the lookup key. The value must remain the exact path
        // used by the backing container (notably for ZIP entries such as
        // `./file.ply`).
        .map(|p| (PathKey::from_path(p), p.clone()))
        .collect()
}

fn checked_lookup_from_paths(paths: &[PathBuf]) -> io::Result<HashMap<PathKey, PathBuf>> {
    let mut lookup: HashMap<PathKey, PathBuf> = HashMap::new();
    for path in paths.iter().filter(|path| is_lookup_path(path)) {
        let key = PathKey::from_path(path);
        if let Some(existing) = lookup.get(&key) {
            return Err(Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Ambiguous ZIP paths normalize to the same name: {} and {}",
                    existing.display(),
                    path.display()
                ),
            ));
        }
        lookup.insert(key, path.clone());
    }
    Ok(lookup)
}

fn zip_error(e: async_zip::error::ZipError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

#[derive(Debug, Error)]
pub enum VfsConstructError {
    #[error("I/O error while constructing BrushVfs.")]
    IoError(#[from] std::io::Error),
    #[error("Got a status page instead of content: \n\n {0}")]
    ReceivedHTML(String),
    #[error("Unknown data type. Only zip and ply files are supported")]
    UnknownDataType,
}

impl BrushVfs {
    pub fn file_count(&self) -> usize {
        self.lookup.len()
    }

    pub fn file_paths(&self) -> impl Iterator<Item = PathBuf> {
        self.lookup.values().cloned()
    }

    pub async fn from_reader(
        mut reader: impl DynRead + 'static,
        name: Option<String>,
    ) -> Result<Self, VfsConstructError> {
        // Small hack to peek some bytes: Read them
        // and add them at the start again.
        let peek = read_at_most(&mut reader, 64).await?;
        let mut reader: Box<dyn DynRead> =
            Box::new(AsyncReadExt::chain(Cursor::new(peek.clone()), reader));

        if peek.starts_with(b"ply") {
            // For single PLY files, keep the reader for streaming
            let mut path = PathBuf::from(name.unwrap_or_default());
            if path.file_name().is_none() {
                path = PathBuf::from("input.ply");
            } else if path
                .extension()
                .is_none_or(|extension| !extension.eq_ignore_ascii_case("ply"))
            {
                path.set_extension("ply");
            }

            Ok(Self {
                // The bytes already identified this as a PLY. Adding the
                // missing extension lets downstream format discovery treat an
                // Android picker name such as `file` as a standalone PLY.
                lookup: HashMap::from([(PathKey::from_path(&path), path)]),
                container: VfsContainer::Streaming {
                    reader: Arc::new(Mutex::new(Some(reader))),
                },
            })
        } else if peek.starts_with(b"PK") {
            let mut zip_reader = ZipFileReader::new(reader.compat());
            let mut entries = HashMap::new();

            while let Some(mut entry) = zip_reader.next_with_entry().await.map_err(zip_error)? {
                if let Ok(filename) = entry.reader().entry().filename().clone().as_str() {
                    let mut data = vec![];
                    let mut reader = entry.reader_mut().compat();
                    reader.read_to_end(&mut data).await?;
                    let path = PathBuf::from(filename);
                    if entries.insert(path.clone(), Arc::new(data)).is_some() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Duplicate ZIP path: {}", path.display()),
                        )
                        .into());
                    }
                    zip_reader = entry.skip().await.map_err(zip_error)?;
                } else {
                    zip_reader = entry.skip().await.map_err(zip_error)?;
                }

                brush_async::yield_now().await;
            }

            let mut path_bufs = entries.keys().cloned().collect::<Vec<_>>();
            path_bufs.sort();
            let lookup = checked_lookup_from_paths(&path_bufs)?;

            Ok(Self {
                lookup,
                container: VfsContainer::InMemory { entries },
            })
        } else if peek.starts_with(b"<!DOCTYPE html>") {
            let mut html = String::new();
            reader.read_to_string(&mut html).await?;
            Err(VfsConstructError::ReceivedHTML(html))
        } else {
            Err(VfsConstructError::UnknownDataType)
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub async fn from_path(dir: &Path) -> Result<Self, VfsConstructError> {
        if dir.is_file() {
            // Construct a reader. This is needed for zip files, as
            // it's not really just a single path.
            let file = tokio::fs::File::open(dir).await?;
            let reader = BufReader::new(file);
            let name = dir.file_name().and_then(|n| n.to_str()).map(String::from);
            Self::from_reader(reader, name).await
        } else {
            // Make a VFS with all files contained in the directory.
            async fn walk_dir(dir: impl AsRef<Path>) -> io::Result<Vec<PathBuf>> {
                let dir = PathBuf::from(dir.as_ref());

                let mut paths = Vec::new();
                let mut stack = vec![dir.clone()];

                while let Some(path) = stack.pop() {
                    let mut read_dir = tokio::fs::read_dir(&path).await?;

                    while let Some(entry) = read_dir.next_entry().await? {
                        let path = entry.path();
                        if path.is_dir() {
                            stack.push(path.clone());
                        } else {
                            let path = path
                                .strip_prefix(dir.clone())
                                .map_err(|_e| io::ErrorKind::InvalidInput)?
                                .to_path_buf();
                            paths.push(path);
                        }

                        brush_async::yield_now().await;
                    }
                }
                Ok(paths)
            }

            let files = walk_dir(dir).await?;
            Ok(Self {
                lookup: lookup_from_paths(&files),
                container: VfsContainer::Directory {
                    base_path: dir.to_path_buf(),
                },
            })
        }
    }

    #[cfg(target_family = "wasm")]
    pub async fn from_directory_handle(
        dir_handle: rrfd::wasm::DirectoryHandle,
    ) -> Result<Self, VfsConstructError> {
        // List all files in the directory
        let paths = dir_handle.list_files().await.map_err(|_e| {
            VfsConstructError::IoError(io::Error::other("Failed to list directory contents"))
        })?;

        Ok(Self {
            lookup: lookup_from_paths(&paths),
            container: VfsContainer::Directory { dir_handle },
        })
    }

    pub fn files_with_extension<'a>(
        &'a self,
        extension: &'a str,
    ) -> impl Iterator<Item = PathBuf> + 'a {
        let extension = extension.to_lowercase();

        self.lookup.values().filter_map(move |path| {
            let ext = path
                .extension()
                .and_then(|ext| ext.to_str())?
                .to_lowercase();
            (ext == extension).then(|| path.clone())
        })
    }

    pub fn files_ending_in<'a>(&'a self, end_path: &str) -> impl Iterator<Item = &'a Path> + 'a {
        let end_keyed = PathKey::from_str(end_path).0;

        self.lookup
            .iter()
            .filter(move |kv| kv.0.0.ends_with(&end_keyed))
            .map(|kv| kv.1.as_path())
    }

    /// Iterate over all files in the VFS.
    pub fn iter_files<'a>(&'a self) -> impl Iterator<Item = &'a Path> + 'a {
        self.lookup.values().map(|path| path.as_path())
    }

    pub async fn reader_at_path(&self, path: &Path) -> io::Result<Box<dyn DynRead>> {
        let key = PathKey::from_path(path);

        let resolved = self.lookup.get(&key).or_else(|| {
            // Datasets (e.g. a NeRFStudio transforms.json) sometimes reference
            // files by absolute path. If we loaded a directory and that path
            // points inside it, strip the directory prefix and resolve it
            // within the VFS. Files outside the VFS are never read.
            let base = PathKey::from_path(&self.base_path()?);
            let rel = key.0.strip_prefix(&base.0)?;
            // Only a match on a path-component boundary counts.
            rel.starts_with('/')
                .then(|| self.lookup.get(&PathKey(rel.to_owned())))
                .flatten()
        });

        let path = resolved.ok_or_else(|| {
            Error::new(
                io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            )
        })?;

        match &self.container {
            VfsContainer::InMemory { entries } => {
                let data = entries.get(path).expect("Unreachable").clone();
                let reader: Box<dyn DynRead> = Box::new(Cursor::new(ArcVec(data)));
                Ok(reader)
            }
            VfsContainer::Streaming { reader } => {
                // Streaming reader can only be consumed once
                let reader: Box<dyn DynRead> = reader
                    .lock()
                    .await
                    .take()
                    .ok_or_else(|| Error::other("Streaming file has already been read"))?;
                Ok(reader)
            }
            #[cfg(not(target_family = "wasm"))]
            VfsContainer::Directory { base_path } => {
                let total_path = base_path.join(path);
                // Higher capacity buffer helps performance
                let file = tokio::io::BufReader::with_capacity(
                    5 * 1024 * 1024,
                    tokio::fs::File::open(total_path).await?,
                );
                let reader: Box<dyn DynRead> = Box::new(file);
                Ok(reader)
            }
            #[cfg(target_family = "wasm")]
            VfsContainer::Directory { dir_handle } => {
                use futures_util::StreamExt;
                use tokio_util::io::StreamReader;
                use wasm_bindgen::JsCast;

                let file = dir_handle.get_file(path).await.map_err(|_e| {
                    Error::new(
                        io::ErrorKind::NotFound,
                        format!("File not found: {}", path.display()),
                    )
                })?;

                let stream = wasm_streams::ReadableStream::from_raw(file.stream())
                    .into_stream()
                    .map(|result| {
                        result
                            .map_err(|e| Error::other(format!("{e:?}")))
                            .and_then(|chunk| {
                                let array =
                                    chunk.dyn_into::<js_sys::Uint8Array>().map_err(|_e| {
                                        Error::new(io::ErrorKind::InvalidData, "Invalid chunk")
                                    })?;
                                let mut data = vec![0u8; array.length() as usize];
                                array.copy_to(&mut data);
                                Ok(tokio_util::bytes::Bytes::from(data))
                            })
                    });

                let reader: Box<dyn DynRead> = Box::new(BufReader::new(StreamReader::new(stream)));
                Ok(reader)
            }
        }
    }

    pub fn empty() -> Self {
        Self {
            lookup: HashMap::new(),
            container: VfsContainer::InMemory {
                entries: HashMap::new(),
            },
        }
    }

    /// Create a test VFS from file paths with empty content.
    #[doc(hidden)]
    pub fn create_test_vfs(paths: Vec<PathBuf>) -> Self {
        let lookup = lookup_from_paths(&paths);

        let entries = paths
            .into_iter()
            .filter(|p| p.extension().is_some())
            .map(|p| (p, Arc::new(Vec::new())))
            .collect();

        Self {
            lookup,
            container: VfsContainer::InMemory { entries },
        }
    }

    pub fn base_path(&self) -> Option<PathBuf> {
        match &self.container {
            VfsContainer::InMemory { .. } => None,
            VfsContainer::Streaming { .. } => None,
            #[cfg(not(target_family = "wasm"))]
            VfsContainer::Directory { base_path } => Some(base_path.clone()),
            #[cfg(target_family = "wasm")]
            VfsContainer::Directory { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use tokio::io::{AsyncRead, AsyncReadExt, ReadBuf};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    struct OneByteReader {
        data: Vec<u8>,
        offset: usize,
    }

    impl OneByteReader {
        fn new(data: impl Into<Vec<u8>>) -> Self {
            Self {
                data: data.into(),
                offset: 0,
            }
        }
    }

    impl AsyncRead for OneByteReader {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            if self.offset < self.data.len() && buf.remaining() > 0 {
                let byte = self.data[self.offset];
                self.offset += 1;
                buf.put_slice(&[byte]);
            }
            Poll::Ready(Ok(()))
        }
    }

    fn fragmented_reader(data: impl Into<Vec<u8>>) -> BufReader<OneByteReader> {
        BufReader::with_capacity(1, OneByteReader::new(data))
    }

    async fn create_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        use async_zip::base::write::ZipFileWriter;
        use async_zip::{Compression, ZipEntryBuilder};

        let mut buffer = Vec::new();
        let mut writer = ZipFileWriter::new(&mut buffer);

        for (name, data) in entries {
            let entry = ZipEntryBuilder::new((*name).into(), Compression::Stored);
            writer.write_entry_whole(entry, data).await.unwrap();
        }

        writer.close().await.unwrap();
        buffer
    }

    async fn create_test_zip() -> Vec<u8> {
        create_zip(&[
            ("test.txt", b"hello world"),
            ("data.json", b"{\"key\": \"value\"}"),
        ])
        .await
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_zip_vfs_workflow() {
        let zip_data = create_test_zip().await;
        let vfs = BrushVfs::from_reader(Cursor::new(zip_data), None)
            .await
            .unwrap();
        assert_eq!(vfs.file_count(), 2);

        let txt_files: Vec<_> = vfs.files_with_extension("txt").collect();
        assert_eq!(txt_files.len(), 1);

        let json_files: Vec<_> = vfs.files_with_extension("json").collect();
        assert_eq!(json_files.len(), 1);

        let mut content = String::new();
        vfs.reader_at_path(&txt_files[0])
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "hello world");

        // Test JSON file
        let mut content = String::new();
        vfs.reader_at_path(&json_files[0])
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "{\"key\": \"value\"}");

        // Test case-insensitive access
        let mut content = String::new();
        vfs.reader_at_path(Path::new("TEST.TXT"))
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "hello world");

        assert!(
            vfs.reader_at_path(Path::new("nonexistent.txt"))
                .await
                .is_err()
        );
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_normalized_zip_path_reads_original_entry() {
        let zip_data = create_zip(&[("./models/foo.ply", b"ply data")]).await;
        let vfs = BrushVfs::from_reader(Cursor::new(zip_data), None)
            .await
            .unwrap();

        let mut content = String::new();
        vfs.reader_at_path(Path::new("models/foo.ply"))
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "ply data");
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_ambiguous_normalized_zip_paths_are_rejected() {
        let zip_data = create_zip(&[("./foo.ply", b"first"), ("foo.ply", b"second")]).await;
        let result = BrushVfs::from_reader(Cursor::new(zip_data), None).await;

        assert!(matches!(
            result,
            Err(VfsConstructError::IoError(error))
                if error.kind() == io::ErrorKind::InvalidData
        ));
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_duplicate_zip_paths_are_rejected() {
        let zip_data = create_zip(&[("foo.ply", b"first"), ("foo.ply", b"second")]).await;
        let result = BrushVfs::from_reader(Cursor::new(zip_data), None).await;

        assert!(matches!(
            result,
            Err(VfsConstructError::IoError(error))
                if error.kind() == io::ErrorKind::InvalidData
        ));
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_extensionless_streaming_ply_is_visible() {
        let ply = b"ply\nformat ascii 1.0\nend_header\n";
        let vfs = BrushVfs::from_reader(Cursor::new(ply), Some("file".to_owned()))
            .await
            .unwrap();

        assert_eq!(vfs.file_count(), 1);
        assert_eq!(vfs.iter_files().next(), Some(Path::new("file.ply")));
        assert_eq!(vfs.files_with_extension("ply").count(), 1);

        let mut content = Vec::new();
        vfs.reader_at_path(Path::new("file.ply"))
            .await
            .unwrap()
            .read_to_end(&mut content)
            .await
            .unwrap();
        assert_eq!(content, ply);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_empty_ply_names_get_usable_filenames() {
        let ply = b"ply\nformat ascii 1.0\nend_header\n";

        for (name, expected) in [("", "input.ply"), ("file.", "file.ply")] {
            let vfs = BrushVfs::from_reader(Cursor::new(ply), Some(name.to_owned()))
                .await
                .unwrap();

            assert_eq!(vfs.iter_files().next(), Some(Path::new(expected)));
            assert_eq!(vfs.files_with_extension("ply").count(), 1);
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_content_confirmed_ply_replaces_misleading_extension() {
        let ply = b"ply\nformat ascii 1.0\nend_header\n";

        for (name, expected) in [("scan.bin", "scan.ply"), ("scan.ply?token", "scan.ply")] {
            let vfs = BrushVfs::from_reader(Cursor::new(ply), Some(name.to_owned()))
                .await
                .unwrap();

            assert_eq!(vfs.iter_files().next(), Some(Path::new(expected)));
            assert_eq!(vfs.files_with_extension("ply").count(), 1);
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_fragmented_ply_format_detection() {
        let ply = b"ply\nformat ascii 1.0\nend_header\nvertex data";
        let vfs = BrushVfs::from_reader(fragmented_reader(ply), None)
            .await
            .unwrap();

        let mut content = Vec::new();
        vfs.reader_at_path(Path::new("input.ply"))
            .await
            .unwrap()
            .read_to_end(&mut content)
            .await
            .unwrap();
        assert_eq!(content, ply);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_fragmented_zip_format_detection() {
        let zip_data = create_test_zip().await;
        let vfs = BrushVfs::from_reader(fragmented_reader(zip_data), None)
            .await
            .unwrap();

        assert_eq!(vfs.file_count(), 2);
        let mut content = String::new();
        vfs.reader_at_path(Path::new("test.txt"))
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "hello world");
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_absolute_path_resolves_within_directory() {
        // Datasets sometimes reference files by absolute path (e.g. a
        // NeRFStudio transforms.json). When the directory was loaded as a
        // VFS, that absolute path should resolve to the file inside it.
        let dir = std::env::temp_dir().join("brush_vfs_abs_test_dir");
        tokio::fs::create_dir_all(dir.join("images")).await.unwrap();
        tokio::fs::write(dir.join("images/cam.png"), b"image content")
            .await
            .unwrap();

        let vfs = BrushVfs::from_path(&dir).await.unwrap();

        // Sanity: relative access works.
        let mut content = String::new();
        vfs.reader_at_path(Path::new("images/cam.png"))
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "image content");

        // The absolute path (dir is absolute) resolves to the same file by
        // stripping the loaded-directory prefix.
        let abs = dir.join("images/cam.png");
        assert!(abs.is_absolute());
        let mut content = String::new();
        vfs.reader_at_path(&abs)
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "image content");

        // An absolute path outside the loaded directory is NOT read.
        let outside = std::env::temp_dir().join("brush_vfs_outside_secret.txt");
        tokio::fs::write(&outside, b"secret").await.unwrap();
        assert!(vfs.reader_at_path(&outside).await.is_err());

        // An empty VFS has no prefix, so absolute paths never resolve.
        assert!(BrushVfs::empty().reader_at_path(&abs).await.is_err());

        tokio::fs::remove_file(&outside).await.unwrap();
        tokio::fs::remove_dir_all(&dir).await.unwrap();
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_format_detection_and_errors() {
        // Test PLY format
        let vfs = BrushVfs::from_reader(
            Cursor::new(b"ply\nformat ascii 1.0\nend_header\nvertex data"),
            None,
        )
        .await
        .unwrap();
        let mut content = String::new();
        vfs.reader_at_path(Path::new("input.ply"))
            .await
            .unwrap()
            .read_to_string(&mut content)
            .await
            .unwrap();
        assert_eq!(content, "ply\nformat ascii 1.0\nend_header\nvertex data");

        // Test error cases
        assert!(matches!(
            BrushVfs::from_reader(Cursor::new(b"unknown"), None).await,
            Err(VfsConstructError::UnknownDataType)
        ));
        assert!(matches!(
            BrushVfs::from_reader(Cursor::new(b"<!DOCTYPE html>"), None).await,
            Err(VfsConstructError::ReceivedHTML(_))
        ));
    }
}
