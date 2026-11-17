# Using Clario

Clario turns a meeting recording or a pasted transcript into a summary, tasks, and decisions. This page describes what the application actually does.

## Navigation

The top bar, after you sign in, has Dashboard, Tasks, Recordings, Trends, Upload, Integrations, Assistant, and Start meeting. Search meetings is the box in that bar. Log out is at the end of the bar. On a narrow screen the same links are behind the menu button.

Dashboard lists your processed meetings. Tasks is the shared task board. Recordings lists saved microphone tracks you are allowed to play. Trends compares recent meetings you own. Upload accepts an audio file or a pasted transcript. Integrations is where Google Meet is connected or disconnected. Assistant answers questions about Clario and about meetings you can already open.

## Start a meeting

Start meeting opens a page where the host creates a join link and can email that link. People who are signed in join the room from the link. The host admits guests from the waiting room. The microphone and camera are not recorded until that person checks the consent box and presses Record. Each browser records that person's own microphone. The host ends the meeting once, and Clario then transcribes the saved tracks.

## Upload a transcript

Upload asks for each attendee's name and email before it accepts an audio file or text pasted into the transcript box. At least one person is required, unless the people were already imported from Google Meet. Tasks are sent to those addresses when the spoken name or the email matches. On the free hosting plan, audio is accepted only when a laptop transcription service is configured. Pasted text still runs. After upload, the meeting page stays on a progress view until the status is completed or failed.

## Transcription

Speech is transcribed with the configured speech model, or forwarded to the laptop transcription service when that service URL and token are both set. If transcription fails, the meeting page says the recording was kept and can be retried. The owner can start transcription again from that meeting. The saved audio is not deleted by a retry. When the laptop service is offline or returns an error, paste the transcript on Upload instead.

## Tasks

Tasks extracted from a meeting show the assignee, due date, priority, and status. Status values are pending, started, doing, and completed. On the task board, only the assignee's account can change that status. Other people can see the status. A name that matches more than one person waits for the host or the assigner to confirm before mail is sent. The assistant can look tasks up. It cannot mark, edit, or delete them.

## Decisions and evidence

Decisions are saved from the transcript and shown on the meeting result page with the spoken line. When a task or decision has a speaker and a timestamp, the result page can open that line and play the saved recording from that time. A missing time stays empty.

## Reports

The meeting result page shows the transcript, the summary, tasks, decisions, the uploaded meeting audio, and the charts generated for that meeting. Trends shows how recent meetings you own compare. Search finds a meeting you own when the query is the meeting title, or an attendee's name or email. A word from the transcript is not enough.

## Google Meet

Integrations connects a Google account with OAuth. After it is connected, Clario can list recent Google Meet conferences and import participants into a meeting you own. Disconnect is on the same Integrations page. Mail can be sent as the signed-in host after that person grants Gmail send.
