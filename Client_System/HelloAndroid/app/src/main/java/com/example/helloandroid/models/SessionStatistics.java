package com.example.helloandroid.models;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.util.ArrayList;
import java.util.List;

public class SessionStatistics {
    private String sessionId;
    private String startDate;
    private String endDate;
    private StateStatistics stateStatistics;
    private List<DetectionImage> images;

    public SessionStatistics(JSONObject json) throws JSONException {
        this.sessionId = json.getString("session_id");
        this.startDate = json.getString("session_start");
        this.endDate = json.getString("session_end");

        JSONObject statsJson = json.getJSONObject("state_statistics");
        this.stateStatistics = new StateStatistics(
                statsJson.getInt("long_absence_count"),
                statsJson.getInt("eyes_closed_count"),
                statsJson.getInt("normal_count")
        );

        this.images = new ArrayList<>();
        JSONArray imagesJson = json.getJSONArray("images");
        for (int i = 0; i < imagesJson.length(); i++) {
            images.add(new DetectionImage(imagesJson.getJSONObject(i)));
        }
    }

    public String getSessionId() { return sessionId; }
    public String getStartDate() { return startDate; }
    public String getEndDate() { return endDate; }
    public StateStatistics getStateStatistics() { return stateStatistics; }
    public List<DetectionImage> getImages() { return images; }
}