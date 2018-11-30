package com.example.android.tflitecamerademo;

import android.app.Activity;
import android.content.ContentUris;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.Image;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.design.internal.BottomNavigationItemView;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.w3c.dom.Text;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class ResultActivity extends Activity {

    private ImageView iv_original; //显示原始图片
    private ImageView iv_result;  //显示处理后的图像
    private ImageView iv_back; //返回按钮
    private ImageView iv_save; //保存按钮
    private TextView time_ex;//显示执行时间
    private ImageSegmentation classifier; // Segment an image
    private static final String TAG = "TfLiteCameraDemo";

    private Bitmap ori_pic;
    private Bitmap result_pic;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.result);

        try {
            // create either a new ImageSegmentation classifier
            classifier = new ImageSegmentationFloatUnet(this);
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize an image classifier.", e);
        }

        iv_original = (ImageView)findViewById(R.id.orgin_pic);
        iv_result = (ImageView)findViewById(R.id.result_pic);
        time_ex = (TextView) findViewById(R.id.time_ex);

        iv_back = (ImageView)findViewById(R.id.iv_back);
        iv_back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), Camera2BasicFragment.class);
                startActivity(intent);
            }
        });
        iv_save = (ImageView)findViewById(R.id.iv_save);
        iv_save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-DD-HH-mm-ss");
                try {
                    save2phone(ori_pic, sdf.format(new Date()).toString() + "-ori");
                    save2phone(result_pic, sdf.format(new Date()).toString() + "-result");

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
        //从intent中获取数据
        String chose = (String) getIntent().getExtras().get("chose");
        if("false".equals(chose)){
//            Uri uri = (Uri)getIntent().getExtras().get("ori");
//            Bitmap bitmap = getBitmapFromUri(uri);
//            ori_pic = reverseBitmap(bitmap);
//            iv_original.setImageURI(uri);
            ori_pic = (Bitmap)getIntent().getExtras().get("ori");
            ori_pic = reverseBitmap(ori_pic);
            iv_original.setImageBitmap(ori_pic);
            applyPicture();
//            iv_result.setImageResource(R.drawable.resu);
        }else{
            openAlbum();
        }
    }

   private Bitmap getBitmapFromUri(Uri uri)
    {
     try
     {
      Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
      return bitmap;
     }
     catch (Exception e)
      {
          Log.e("[Android]", e.getMessage());
          Log.e("[Android]", "Ä¿ÂŒÎª£º" + uri);
          e.printStackTrace();
          return null;
         }
     }
    //-----------------------------选择图片----------------------------------------------------------------

    private static final int CHOOSE_PHOTO = 1;
    private void openAlbum() {
        Intent intent = new Intent("android.intent.action.GET_CONTENT");
        intent.setType("image/*");
        startActivityForResult(intent, CHOOSE_PHOTO);//打开相册
//        startActivity(intent);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case 1:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    openAlbum();
                } else {
                    Toast.makeText(getApplicationContext(), "You denied the permission", Toast.LENGTH_SHORT).show();
                }
                break;
            default:
                break;
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode) {
            case CHOOSE_PHOTO:
                if (resultCode == RESULT_OK) {

                    //判断手机系统版本号
                    if (Build.VERSION.SDK_INT >= 19) {
                        //4.4及以上系统使用这个方法处理图片
                        handleImageOnKitKat(data);
                    } else {
                        //4.4以下系统使用这个放出处理图片
                        handleImageBeforeKitKat(data);
                    }
                }else{
                    Intent intent = new Intent(getApplicationContext(), Camera2BasicFragment.class);
                    startActivity(intent);
                }
                break;

            default:
                break;
        }
    }

    private void handleImageOnKitKat(Intent data) {
        String imagePath = null;
        Uri uri = data.getData();
        if (DocumentsContract.isDocumentUri(getApplicationContext(), uri)) {
            //如果是document类型的Uri,则通过document id处理
            String docId = DocumentsContract.getDocumentId(uri);
            if ("com.android.providers.media.documents".equals(uri.getAuthority())) {
                String id = docId.split(":")[1];//解析出数字格式的id
                String selection = MediaStore.Images.Media._ID + "=" + id;
                imagePath = getImagePath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, selection);
            } else if ("com.android.providers.downloads.documents".equals(uri.getAuthority())) {
                Uri contentUri = ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"), Long.valueOf(docId));
                imagePath = getImagePath(contentUri, null);
            }
        } else if ("content".equalsIgnoreCase(uri.getScheme())) {
            //如果是content类型的Uri，则使用普通方式处理
            imagePath = getImagePath(uri, null);
        } else if ("file".equalsIgnoreCase(uri.getScheme())) {
            //如果是file类型的Uri，直接获取图片路径即可
            imagePath = uri.getPath();
        }
        diaplayImage(imagePath);
    }

    private void handleImageBeforeKitKat(Intent data){
        Uri uri = data.getData();
        String imagePath = getImagePath(uri,null);
        diaplayImage(imagePath);
    }

    private String getImagePath(Uri uri,String selection){
        String path = null;
        //通过Uri和selection来获取真实的图片路径
        Cursor cursor = getApplicationContext().getContentResolver().query(uri,null,selection,null,null);
        if (cursor != null){
            if (cursor.moveToFirst()){
                path = cursor.getString(cursor.getColumnIndex(MediaStore.Images.Media.DATA));
            }
            cursor.close();
        }
        return path;
    }


    private void diaplayImage(String imagePath) {
        Bitmap bitmap = null;
        if (imagePath != null) {
            bitmap = BitmapFactory.decodeFile(imagePath);
            bitmap = reverseBitmap(bitmap);
            iv_original.setImageBitmap(bitmap);
            ori_pic = bitmap;
            applyPicture();
//            iv_result.setImageResource(R.drawable.resu);
        } else {
            Intent intent = new Intent(getApplicationContext(), Camera2BasicFragment.class);
            startActivity(intent);
        }
    }

    private Bitmap reverseBitmap(Bitmap bitmap){

        // rotate the image
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Matrix matrix = new Matrix();
        if (width < height) {    // assert width >= height
            matrix.setRotate(90);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
            int tmp = width;
            width = height;
            height = tmp;
        }

        // cut and rotate the image.
        int stw, sth;           // start points for cutting
        int cutw, cuth;         // width and height after cutting
        if (width / height > 1.5) {
            cuth = height;
            cutw = (int) (height * 1.5);
            sth = 0;
            stw = (width - cutw) / 2;
        }
        else {
            cutw = width;
            cuth = (int) (width / 1.5);
            stw = 0;
            sth = (height - cuth) / 2;
        }
        bitmap = Bitmap.createBitmap(bitmap, stw, sth, cutw, cuth, null, true);
        bitmap = Bitmap.createScaledBitmap(bitmap, classifier.getImageSizeY(), classifier.getImageSizeX(), true);

        return bitmap;
    }

    public void save2phone(Bitmap bmp, String picName) throws IOException {
        if(bmp == null) return;
        String fileName = null;
        //系统相册目录
        String galleryPath= Environment.getExternalStorageDirectory()
                + File.separator + Environment.DIRECTORY_DCIM
                +File.separator+"Camera"+File.separator;

        // 声明文件对象
        File file = null;
        // 声明输出流
        FileOutputStream outStream = null;

        try {
            // 如果有目标文件，直接获得文件对象，否则创建一个以filename为名称的文件
            file = new File(galleryPath, picName + ".jpg");

            // 获得文件相对路径
            fileName = file.toString();
            // 获得输出流，如果文件中有内容，追加内容
            outStream = new FileOutputStream(fileName);
            if (null != outStream) {
                bmp.compress(Bitmap.CompressFormat.JPEG, 90, outStream);
            }
        } catch (Exception e) {
            e.getStackTrace();
        }finally{
            if(outStream != null){
                outStream.close();
            }
        }
        MediaStore.Images.Media.insertImage(ResultActivity.this.getContentResolver(), bmp, fileName, null);
        Intent intent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        Uri uri = Uri.fromFile(file);
        intent.setData(uri);
        ResultActivity.this.sendBroadcast(intent);
        Toast.makeText(ResultActivity.this, "保存成功", Toast.LENGTH_SHORT).show();
    }

    //处理具体的图像处理
    private void applyPicture() {
        // Here's where the magic happens!!!
//        long startTime = SystemClock.uptimeMillis();
        //代码写在这里
        long duration = executer();
//        long endTime = SystemClock.uptimeMillis();
//        long duration = endTime - startTime;
        time_ex.setText("执行时间：" + duration + "ms");
    }

    //处理逻辑
    private long executer() {
        // create a bitmap to hold the result.
        result_pic = Bitmap.createBitmap(classifier.getImageSizeY(), classifier.getImageSizeX(), Bitmap.Config.ARGB_8888);

        // process
        long duration = classifier.segmentFrame(ori_pic, result_pic);

        //显示最后处理的图像  bm为bitmap类的对象
        iv_result.setImageBitmap(result_pic);
//        result_pic = bm;
        return duration;
    }
}
