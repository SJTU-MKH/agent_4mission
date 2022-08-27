// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements Runnable {
    private int mImageIndex = 0;
    private String[] mTestImages = {"main1.jpg", "main2.png", "main3.png", "main4.jpg"};

    private TextView textView;
    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    private Module graphModule = null;
    private Bitmap graphBitmap = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;
    private Module colorModule = null;
    private Bitmap colorBitmap = null;
    String[] qr_content;
    // 模型写入缓存
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    // load lib
    static {
        System.loadLibrary("objectdetection");
    }

    /*
        权限申请，　
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R || Environment.isExternalStorageManager()) {

        } else {
            Intent intent = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
            startActivity(intent);
        }

        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        textView = findViewById(R.id.debug);
        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);

        textView.setText(initQr());
        final Button buttonTest = findViewById(R.id.testButton);
        buttonTest.setText(("Test Image 1/4"));
        buttonTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestImages.length;
                buttonTest.setText(String.format("Text Image %d/%d", mImageIndex + 1, mTestImages.length));

                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });

        // 选择事件: 相册、拍照、取消
        final Button buttonSelect = findViewById(R.id.selectButton);
        buttonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);

                final CharSequence[] options = { "Choose from Photos", "Take Picture", "Cancel" };
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("New Test Image");

                builder.setItems(options, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int item) {
                        if (options[item].equals("Take Picture")) {
                            Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                            startActivityForResult(takePicture, 0);
                        }
                        else if (options[item].equals("Choose from Photos")) {
                            Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                            startActivityForResult(pickPhoto , 1);
                        }
                        else if (options[item].equals("Cancel")) {
                            dialog.dismiss();
                        }
                    }
                });
                builder.show();
            }
        });

        final Button buttonLive = findViewById(R.id.liveButton);
        buttonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
              final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
              startActivity(intent);
            }
        });

        mButtonDetect = findViewById(R.id.detectButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonDetect.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonDetect.setText(getString(R.string.run_model));

                mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.mInputWidth;
//                mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.mInputHeight;
                mImgScaleY = (float)mBitmap.getHeight() / 180; //实际缩放
                // ImageView (1080, 1080)   mbitMap(320, 320)
                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mImageView.getWidth() / mBitmap.getWidth() : (float)mImageView.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mImageView.getHeight() / mBitmap.getHeight() : (float)mImageView.getWidth() / mBitmap.getWidth());
//                mIvScaleX = (float)mImageView.getWidth() / mBitmap.getWidth();
//                mIvScaleY  = (float)mImageView.getHeight() / mBitmap.getHeight();
                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (mImageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });
        // 模型加载
        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "best.ptl"));
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("mainClasses.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);
            // subgraph
            graphModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "best_graph.ptl"));
            colorModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "color.ptl"));
//            BufferedReader br_graph = new BufferedReader(new InputStreamReader(getAssets().open("graphClasses.txt")));
//            String line_graph;
//            List<String> classes_graph = new ArrayList<>();
//            while ((line_graph = br_graph.readLine()) != null) {
//                classes_graph.add(line_graph);
//            }
//            PrePostProcessor.graphClasses = new String[classes_graph.size()];
//            classes_graph.toArray(PrePostProcessor.graphClasses);
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
    }
    // 选择事件
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
//                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
//                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }
    // 高度上下零填充70行
    private Tensor pad(Tensor x){
        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * PrePostProcessor.mInputWidth * PrePostProcessor.mInputHeight);
        float value = (float) (114.0/255.0);

        for(int i=0; i<320*320*3; i++)
            floatBuffer.put(i, value);
        int offset = 70 * 320;
        float[] rawdata = x.getDataAsFloatArray();
        int[] new_offset_rgb = {0, 102400, 204800};
        int[] old_offset_rgb = {0, 57600, 115200};
        for(int i = 0; i<180*320; i++){
            for(int j=0;j<3;j++) {
                floatBuffer.put(new_offset_rgb[j]+offset+i, rawdata[old_offset_rgb[j]+i]);
            }
        }
        return Tensor.fromBlob(floatBuffer, new long[]{1, 3, PrePostProcessor.mInputHeight, PrePostProcessor.mInputWidth});
    }
    // 上下pad
    private Tensor pad(Tensor x, int up, int height){
        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * PrePostProcessor.mInputWidth * PrePostProcessor.mInputHeight);
        float value = (float) (114.0/255.0);

        for(int i=0; i<320*320*3; i++)
            floatBuffer.put(i, value);
        int offset = up * 320;
        float[] rawdata = x.getDataAsFloatArray();
        int[] new_offset_rgb = {0, 102400, 204800};
        int[] old_offset_rgb = {0, height*320, height*320*2};
        for(int i = 0; i<height*320; i++){
            for(int j=0;j<3;j++) {
                floatBuffer.put(new_offset_rgb[j]+offset+i, rawdata[old_offset_rgb[j]+i]);
            }
        }
        return Tensor.fromBlob(floatBuffer, new long[]{1, 3, PrePostProcessor.mInputHeight, PrePostProcessor.mInputWidth});
    }



    @Override
    public void run() {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, 180, true);
//        Bitmap.create
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
        final Tensor padTensor = pad(inputTensor);
        IValue[] outputTuple = mModule.forward(IValue.from(padTensor)).toTuple();
        final Tensor outputTensor = outputTuple[0].toTensor();
        final float[] outputs = outputTensor.getDataAsFloatArray();
        final ArrayList<Result> results =  PrePostProcessor.outputsToNMSPredictions(outputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);

        String output = "";
        int qrcode_flag = 0;
        int plate_flag = 0;
        int graph_flag = 0;

        if (results.size()>0)
        {
            for(int i=0; i<results.size(); i++)
                if(results.get(i).classIndex==9)
                {
                    qrcode_flag = 1;
                    qr_content = opBitmap2(resizedBitmap, Bitmap.Config.ARGB_8888);
                    for(int cc=0 ; cc<5; cc++){
                        if(qr_content[cc].length() > 0 ){
                            output = output + qr_content[cc];
                        }
                    }
                    break;
                }
            if(qrcode_flag == 0){
                for(int i=0; i<results.size(); i++)
                    if(results.get(i).classIndex==7 || results.get(i).classIndex==8)
                    {
                        plate_flag = 1;
                        // plate_recognize
                        output = "plate";
                        break;
                    }
            }
            String graph_cls[] = {"圆", "菱形", "五星", "三角", "矩形"};
            String color[] = {"红", "绿", "蓝", "黄", "品红", "青", "黑", "白"};
            if(qrcode_flag == 0 && plate_flag==0){
                for(int i=0; i<results.size(); i++)
                    if(results.get(i).classIndex==6)
                    {
                        output = "";
                        Rect rect = results.get(i).raw_rect;
                        graph_flag = 1;
                        int x_ = rect.left, y_ = rect.top, width_ = rect.right-rect.left, height_ = rect.bottom-rect.top;

                        if(x_<0)
                            x_ = 0;
                        if(y_<0)
                            y_ = 0;
                        if(width_+x_>mBitmap.getWidth())
                            width_ = mBitmap.getWidth() - x_;
                        if(height_+y_>mBitmap.getHeight())
                            height_ = mBitmap.getHeight() - y_;

                        graphBitmap = Bitmap.createBitmap(mBitmap, x_, y_, width_, height_);

                        float ratio = (float)320/width_;
                        int newheight_ = (int) (ratio*height_);
                        int tmp_flag = 0;
                        if(newheight_ > 320){
                            newheight_ = 320;
                            tmp_flag = 1;
                        }

                        Bitmap tmp_resizedBitmap = Bitmap.createScaledBitmap(graphBitmap, 320, newheight_, true);
                        final Tensor inputTensor_ = TensorImageUtils.bitmapToFloat32Tensor(tmp_resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);

                        final Tensor padTensor_;
                        int pad_up;
                        if (tmp_flag==1){
                            pad_up = 0;
                            padTensor_ = inputTensor_;

                        }
                        else{
                            pad_up = (int)(320 - newheight_)/2;
                            padTensor_ = pad(inputTensor_, pad_up, newheight_);
                        }

                        IValue[] outputTuple_ = graphModule.forward(IValue.from(padTensor_)).toTuple();
                        final Tensor outputTensor_ = outputTuple_[0].toTensor();
                        final float[] outputs_ = outputTensor_.getDataAsFloatArray();

                        final ArrayList<Result> results_ =  PrePostProcessor.outputsToNMSPredictions_graph(outputs_, mImgScaleX, mImgScaleY, mStartX, mStartY, x_, y_, ratio, pad_up);
                        for(int i_=0; i_<results_.size();i_++)
                        {
                            Result tmp = results_.get(i_);
                            Rect tmp_rect = tmp.raw_rect;
                            int x_tmp = tmp_rect.left, y_tmp = tmp_rect.top, width_tmp = tmp_rect.right-tmp_rect.left, height_tmp = tmp_rect.bottom-tmp_rect.top;

                            if(x_tmp<0)
                                x_tmp = 0;
                            if(y_tmp<0)
                                y_tmp = 0;
                            if(width_tmp+x_tmp>graphBitmap.getWidth())
                                width_tmp = graphBitmap.getWidth() - x_tmp;
                            if(height_tmp+y_tmp>graphBitmap.getHeight())
                                height_tmp = graphBitmap.getHeight() - y_tmp;

                            Bitmap tmp_sub = Bitmap.createBitmap(graphBitmap, x_tmp, y_tmp, width_tmp, height_tmp);
                            colorBitmap = Bitmap.createScaledBitmap(tmp_sub, 224, 224, true);
                            final Tensor inputTensor_color = TensorImageUtils.bitmapToFloat32Tensor(colorBitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

                            final Tensor outputTensor_color = colorModule.forward(IValue.from(inputTensor_color)).toTensor();
                            final float[] color_scores = outputTensor_color.getDataAsFloatArray();

                            // searching for the index with maximum score
                            float maxScore = -Float.MAX_VALUE;
                            int maxScoreIdx = -1;
                            for (int k = 0; k < color_scores.length; k++) {
                                if (color_scores[k] > maxScore) {
                                    maxScore = color_scores[k];
                                    maxScoreIdx = k;
                                }
                            }

                            output = output + " " + color[maxScoreIdx] + "-"+ graph_cls[tmp.classIndex];
                        }
                        break;
                    }
            }
            if(qrcode_flag == 0 && plate_flag==0 && graph_flag ==0){
                for(int i=0; i<results.size(); i++)
                {
                    output = "traffic";
                }
            }
        }
        else
            output = "none";


        // 更新所有控件
        String finalOutput = output;
        runOnUiThread(() -> {
            mButtonDetect.setEnabled(true);
            mButtonDetect.setText(getString(R.string.detect));
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            mResultView.setResults(results);
            mResultView.invalidate();
            mResultView.setVisibility(View.VISIBLE);
            textView.setText(finalOutput);
        });
    }

    public native String[] opBitmap2(Bitmap bitmap, Bitmap.Config argb8888);
    public native String opBitmap(Bitmap bitmap, Bitmap.Config argb8888);
    public native String initQr();
}
