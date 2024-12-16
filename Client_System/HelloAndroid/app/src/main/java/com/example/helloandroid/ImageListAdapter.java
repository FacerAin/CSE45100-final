package com.example.helloandroid;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.example.helloandroid.models.DetectionImage;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;

public class ImageListAdapter extends RecyclerView.Adapter<ImageListAdapter.ViewHolder> {
    private static final String TAG = "ImageListAdapter";
    private List<DetectionImage> images;
    private String baseUrl;
    private static final String TOKEN = "b9506731a52a9285af72fe2de435c41dd891d44a";

    public ImageListAdapter(List<DetectionImage> images) {
        this.images = images;
        this.baseUrl = "https://syw5141.pythonanywhere.com";
    }

    public void updateImages(List<DetectionImage> newImages) {
        this.images = newImages;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_image, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        DetectionImage image = images.get(position);

        holder.itemView.setBackgroundColor(image.getStateColorWithAlpha());

        // 이미지 로드
        String imageUrl = image.getImageUrl();
        if (imageUrl != null && !imageUrl.isEmpty()) {
            // 전체 URL 구성
            String fullUrl = baseUrl + "/media/" + imageUrl;
            Log.d(TAG, "Loading image from URL: " + fullUrl);
            new LoadImageTask(holder.imageView).execute(fullUrl);
        }

        holder.textTitle.setText(image.getTitle());
        holder.textState.setText(image.getState());

        String date = image.getPublishedDate();
        if (date != null && date.length() >= 19) {
            holder.textDate.setText(date.substring(0, 19).replace('T', ' '));
        }
    }

    @Override
    public int getItemCount() {
        return images != null ? images.size() : 0;
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        ImageView imageView;
        TextView textTitle;
        TextView textState;
        TextView textDate;

        ViewHolder(View itemView) {
            super(itemView);
            imageView = itemView.findViewById(R.id.imageView);
            textTitle = itemView.findViewById(R.id.textTitle);
            textState = itemView.findViewById(R.id.textState);
            textDate = itemView.findViewById(R.id.textDate);
        }
    }

    private static class LoadImageTask extends AsyncTask<String, Void, Bitmap> {
        private final ImageView imageView;

        LoadImageTask(ImageView imageView) {
            this.imageView = imageView;
        }

        @Override
        protected Bitmap doInBackground(String... params) {
            try {
                String urlString = params[0];
                Log.d(TAG, "Attempting to load image from: " + urlString);

                URL url = new URL(urlString);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setRequestProperty("Authorization", "Token " + TOKEN);
                conn.setConnectTimeout(5000);
                conn.setReadTimeout(5000);
                conn.setDoInput(true);

                int responseCode = conn.getResponseCode();
                Log.d(TAG, "HTTP Response Code: " + responseCode);

                if (responseCode == HttpURLConnection.HTTP_OK) {
                    InputStream inputStream = conn.getInputStream();
                    Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                    inputStream.close();
                    return bitmap;
                }
            } catch (Exception e) {
                Log.e(TAG, "Error loading image: " + e.getMessage(), e);
            }
            return null;
        }

        @Override
        protected void onPostExecute(Bitmap result) {
            if (result != null && imageView != null) {
                imageView.setImageBitmap(result);
            }
        }
    }
}