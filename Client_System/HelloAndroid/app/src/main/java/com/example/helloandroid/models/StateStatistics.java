package com.example.helloandroid.models;

public class StateStatistics {
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