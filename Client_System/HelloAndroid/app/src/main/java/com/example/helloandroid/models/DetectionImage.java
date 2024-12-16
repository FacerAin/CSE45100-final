package com.example.helloandroid.models;

import android.graphics.Color;
import org.json.JSONException;
import org.json.JSONObject;

public class DetectionImage {
    private int id;
    private String title;
    private String text;
    private String publishedDate;
    private String sessionId;
    private String author;
    private String imageUrl;  // 추가된 필드

    public DetectionImage(JSONObject json) throws JSONException {
        this.id = json.getInt("id");
        this.title = json.getString("title");
        this.text = json.getString("text");
        this.publishedDate = json.getString("published_date");
        this.sessionId = json.optString("session_id", "");
        this.author = json.optString("author", "");
        this.imageUrl = json.optString("image", "");  // image URL 파싱 추가
    }

    // 기존 getter 메소드들
    public int getId() { return id; }
    public String getTitle() { return title; }
    public String getText() { return text; }
    public String getPublishedDate() { return publishedDate; }
    public String getSessionId() { return sessionId; }
    public String getAuthor() { return author; }
    public String getImageUrl() { return imageUrl; }  // 새로운 getter 메소드

    public String getState() {
        if (text.contains("LONG_ABSENCE")) return "LONG_ABSENCE";
        if (text.contains("EYES_CLOSED_LONG")) return "EYES_CLOSED_LONG";
        return "NORMAL";
    }

    public int getStateColor() {
        switch (getState()) {
            case "LONG_ABSENCE":
                return Color.parseColor("#FFCDD2");  // 빨간색 계열
            case "EYES_CLOSED_LONG":
                return Color.parseColor("#FFE0B2");  // 주황색 계열
            default:
                return Color.parseColor("#C8E6C9");  // 초록색 계열
        }
    }

    // 상태에 따른 배경색 투명도 적용 메소드 추가
    public int getStateColorWithAlpha() {
        return Color.argb(50,  // alpha value for transparency
                Color.red(getStateColor()),
                Color.green(getStateColor()),
                Color.blue(getStateColor()));
    }
}