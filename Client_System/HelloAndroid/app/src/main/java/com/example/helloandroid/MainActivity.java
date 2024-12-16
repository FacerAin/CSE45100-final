package com.example.helloandroid;

import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.example.helloandroid.models.Session;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private TextView textView;
    private String site_url = "https://syw5141.pythonanywhere.com";
    private LoadSessions taskLoadSessions;
    private RecyclerView sessionsRecyclerView;
    private SessionAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = findViewById(R.id.textView);
        sessionsRecyclerView = findViewById(R.id.recyclerView);

        textView.setText("Loading sessions...");

        adapter = new SessionAdapter(new ArrayList<>(), session -> {
            Intent intent = new Intent(MainActivity.this, SessionDetailActivity.class);
            intent.putExtra("session_id", session.getId());
            startActivity(intent);
        });

        sessionsRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        sessionsRecyclerView.setAdapter(adapter);

        loadSessions();
    }

    private void loadSessions() {
        if (taskLoadSessions != null && taskLoadSessions.getStatus() == AsyncTask.Status.RUNNING) {
            taskLoadSessions.cancel(true);
        }
        taskLoadSessions = new LoadSessions();
        taskLoadSessions.execute(site_url + "/api_root/Session/");
    }

    private class LoadSessions extends AsyncTask<String, Integer, List<Session>> {
        private String errorMessage;
        private final String username = "syw5141";
        private final String password = "1234";
        private String token = "b9506731a52a9285af72fe2de435c41dd891d44a";

        @Override
        protected List<Session> doInBackground(String... urls) {
            List<Session> sessionList = new ArrayList<>();
            try {
                // Token 인증 처리
                URL tokenUrl = new URL(site_url + "/api-token-auth/");
                HttpURLConnection tokenConn = (HttpURLConnection) tokenUrl.openConnection();
                tokenConn.setRequestMethod("POST");
                tokenConn.setRequestProperty("Content-Type", "application/json");
                tokenConn.setDoOutput(true);

                JSONObject tokenRequest = new JSONObject();
                tokenRequest.put("username", username);
                tokenRequest.put("password", password);

                try (OutputStream os = tokenConn.getOutputStream()) {
                    byte[] input = tokenRequest.toString().getBytes("utf-8");
                    os.write(input, 0, input.length);
                }

                if (tokenConn.getResponseCode() != HttpURLConnection.HTTP_OK) {
                    errorMessage = "Authentication failed";
                    return null;
                }

                // 세션 목록 가져오기
                URL urlAPI = new URL(urls[0]);
                HttpURLConnection conn = (HttpURLConnection) urlAPI.openConnection();
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

                    JSONArray jsonArray = new JSONArray(result.toString());
                    for (int i = 0; i < jsonArray.length(); i++) {
                        JSONObject sessionJson = jsonArray.getJSONObject(i);
                        Session session = new Session(
                                sessionJson.getInt("id"),
                                sessionJson.getString("session_id"),
                                sessionJson.getString("session_start_date"),
                                sessionJson.getString("session_end_date")
                        );
                        sessionList.add(session);
                    }

                    // 최신 세션이 위로 오도록 정렬
                    Collections.sort(sessionList,
                            (s1, s2) -> s2.getStartDate().compareTo(s1.getStartDate()));
                }
            } catch (Exception e) {
                errorMessage = "Connection error: " + e.getMessage();
                e.printStackTrace();
            }
            return sessionList;
        }

        @Override
        protected void onPostExecute(List<Session> sessions) {
            if (sessions.isEmpty()) {
                textView.setText(errorMessage != null ?
                        errorMessage : "No study sessions found");
            } else {
                textView.setText("Found " + sessions.size() + " study sessions\n" +
                        "Tap a session to view details");
                adapter.updateSessions(sessions);
                adapter.notifyDataSetChanged();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (taskLoadSessions != null) {
            taskLoadSessions.cancel(true);
        }
    }
}