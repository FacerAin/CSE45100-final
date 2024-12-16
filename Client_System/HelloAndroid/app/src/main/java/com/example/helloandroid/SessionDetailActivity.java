package com.example.helloandroid;

import android.os.AsyncTask;
import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.example.helloandroid.models.SessionStatistics;
import org.json.JSONObject;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;

public class SessionDetailActivity extends AppCompatActivity {
    private TextView textSessionPeriod;
    private TextView textAbsenceCount;
    private TextView textEyesClosedCount;
    private TextView textNormalCount;
    private RecyclerView imagesRecyclerView;
    private ImageListAdapter adapter;  // Keep adapter as a field
    private LoadSessionStatistics taskLoadStats;
    private String site_url = "https://syw5141.pythonanywhere.com";
    private String token = "b9506731a52a9285af72fe2de435c41dd891d44a";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_session_detail);

        initializeViews();

        // Initialize with empty adapter
        adapter = new ImageListAdapter(new ArrayList<>());
        imagesRecyclerView.setAdapter(adapter);

        int sessionId = getIntent().getIntExtra("session_id", -1);
        if (sessionId != -1) {
            loadSessionStatistics(sessionId);
        }
    }

    private void initializeViews() {
        textSessionPeriod = findViewById(R.id.textSessionPeriod);
        textAbsenceCount = findViewById(R.id.textAbsenceCount);
        textEyesClosedCount = findViewById(R.id.textEyesClosedCount);
        textNormalCount = findViewById(R.id.textNormalCount);
        imagesRecyclerView = findViewById(R.id.imageRecyclerView);
        imagesRecyclerView.setLayoutManager(new LinearLayoutManager(this));
    }

    private void loadSessionStatistics(int sessionId) {
        taskLoadStats = new LoadSessionStatistics();
        taskLoadStats.execute(site_url + "/api_root/Session/" + sessionId + "/session_statistics/");
    }

    private class LoadSessionStatistics extends AsyncTask<String, Void, SessionStatistics> {
        @Override
        protected SessionStatistics doInBackground(String... urls) {
            try {
                URL url = new URL(urls[0]);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestProperty("Authorization", "Token " + token);
                conn.setRequestMethod("GET");

                if (conn.getResponseCode() == HttpURLConnection.HTTP_OK) {
                    BufferedReader reader = new BufferedReader(
                            new InputStreamReader(conn.getInputStream())
                    );
                    StringBuilder result = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        result.append(line);
                    }

                    return new SessionStatistics(new JSONObject(result.toString()));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }

        @Override
        protected void onPostExecute(SessionStatistics stats) {
            if (stats != null) {
                updateUI(stats);
            }
        }

        private void updateUI(SessionStatistics stats) {
            textSessionPeriod.setText(String.format("Period: %s ~ %s",
                    stats.getStartDate().substring(0, 19).replace('T', ' '),
                    stats.getEndDate().substring(0, 19).replace('T', ' ')));

            textAbsenceCount.setText("Absence Count: " +
                    stats.getStateStatistics().getLongAbsenceCount());
            textEyesClosedCount.setText("Eyes Closed Count: " +
                    stats.getStateStatistics().getEyesClosedCount());
            textNormalCount.setText("Normal Count: " +
                    stats.getStateStatistics().getNormalCount());

            // Update the existing adapter instead of creating a new one
            adapter.updateImages(stats.getImages());
            adapter.notifyDataSetChanged();
        }
    }
}