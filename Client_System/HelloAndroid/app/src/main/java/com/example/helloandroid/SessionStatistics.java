package com.example.helloandroid;

import com.example.helloandroid.models.DetectionImage;

import org.json.JSONArray;
import org.json.JSONObject;
import java.util.ArrayList;
import java.util.List;

public class SessionStatistics {
    private String startDate;
    private String endDate;
    private StateStatistics stateStatistics;
    private List<DetectionImage> images;

    public SessionStatistics(JSONObject json) {
        try {
            startDate = json.getString("session_start");
            endDate = json.getString("session_end");

            JSONObject stats = json.getJSONObject("state_statistics");
            stateStatistics = new StateStatistics(
                    stats.getInt("long_absence_count"),
                    stats.getInt("eyes_closed_count"),
                    stats.getInt("normal_count")
            );

            images = new ArrayList<>();
            JSONArray imagesArray = json.getJSONArray("images");
            for (int i = 0; i < imagesArray.length(); i++) {
                // DetectionImage 객체 생성 시 JSONObject를 직접 전달
                images.add(new DetectionImage(imagesArray.getJSONObject(i)));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String getStartDate() { return startDate; }
    public String getEndDate() { return endDate; }
    public StateStatistics getStateStatistics() { return stateStatistics; }
    public List<DetectionImage> getImages() { return images; }

    public static class StateStatistics {
        private int longAbsenceCount;
        private int eyesClosedCount;
        private int normalCount;

        public StateStatistics(int longAbsenceCount, int eyesClosedCount, int normalCount) {
            this.longAbsenceCount = longAbsenceCount;
            this.eyesClosedCount = eyesClosedCount;
            this.normalCount = normalCount;
        }

        public int getLongAbsenceCount() { return longAbsenceCount; }
        public int getEyesClosedCount() { return eyesClosedCount; }
        public int getNormalCount() { return normalCount; }
    }
}