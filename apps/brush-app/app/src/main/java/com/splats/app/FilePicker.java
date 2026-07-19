package com.splats.app;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;

public class FilePicker {
    @SuppressLint("StaticFieldLeak")
    private static Activity _activity;
    private static final int FIRST_REQUEST_CODE = 0x1000;
    private static final int LAST_REQUEST_CODE = 0x7fff;
    private static native void onFilePickerResult(int requestId, int fd);

    public static void Register(Activity activity) {
        _activity = activity;
    }

    public static void startFilePicker(int requestId) {
        if (!isFilePickerRequest(requestId)) {
            onFilePickerResult(requestId, -1);
            return;
        }

        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        try {
            _activity.startActivityForResult(intent, requestId);
        } catch (RuntimeException e) {
            onFilePickerResult(requestId, -1);
        }
    }

    public static boolean isFilePickerRequest(int requestId) {
        return requestId >= FIRST_REQUEST_CODE && requestId <= LAST_REQUEST_CODE;
    }

    public static void onPicked(int requestId, int fd) {
        onFilePickerResult(requestId, fd);
    }
}
