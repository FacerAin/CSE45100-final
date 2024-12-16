package com.example.helloandroid;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.example.helloandroid.models.Session;
import java.util.List;

public class SessionAdapter extends RecyclerView.Adapter<SessionAdapter.ViewHolder> {
    private List<Session> sessions;
    private OnSessionClickListener listener;

    public void updateSessions(List<Session> sessions) {
        this.sessions = sessions;
        notifyDataSetChanged();
    }

    public interface OnSessionClickListener {
        void onSessionClick(Session session);
    }

    public SessionAdapter(List<Session> sessions, OnSessionClickListener listener) {
        this.sessions = sessions;
        this.listener = listener;
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_session, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(ViewHolder holder, int position) {
        Session session = sessions.get(position);
        holder.textSessionId.setText("Session: " + session.getSessionId());
        holder.textStartDate.setText("Start: " + session.getStartDate().substring(0, 19).replace('T', ' '));
        holder.itemView.setOnClickListener(v -> listener.onSessionClick(session));
    }

    @Override
    public int getItemCount() {
        return sessions.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView textSessionId;
        TextView textStartDate;

        ViewHolder(View itemView) {
            super(itemView);
            textSessionId = itemView.findViewById(R.id.textSessionId);
            textStartDate = itemView.findViewById(R.id.textStartDate);
        }
    }
}