# Jero Autonomy Backlog

During idle time, Jero picks the next unchecked item below, researches it (Groq),
generates a proposed implementation (NVIDIA NIM), and writes it to `proposals/`
for human review. Jero never executes or merges this code automatically.

- [ ] Wake-word detection so Jero only listens after a hotword
- [ ] Conversation memory: persist recent turns and summarize long sessions
- [ ] System resource monitor command ("how much RAM/CPU is in use?")
- [ ] Clipboard read/write handler for quick copy-paste automation
- [ ] Reminder/timer feature with spoken notifications
