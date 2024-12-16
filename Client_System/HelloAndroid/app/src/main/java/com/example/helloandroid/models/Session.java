package com.example.helloandroid.models;

public class Session {
    private int id;
    private String sessionId;
    private String startDate;
    private String endDate;
    private String author;
    private SessionStatistics statistics;

    public Session(int id, String sessionId, String startDate, String endDate) {
        this.id = id;
        this.sessionId = sessionId;
        this.startDate = startDate;
        this.endDate = endDate;
    }

    public int getId() { return id; }
    public String getSessionId() { return sessionId; }
    public String getStartDate() { return startDate; }
    public String getEndDate() { return endDate; }
    public String getAuthor() { return author; }
    public SessionStatistics getStatistics() { return statistics; }

    public void setStatistics(SessionStatistics statistics) {
        this.statistics = statistics;
    }
}