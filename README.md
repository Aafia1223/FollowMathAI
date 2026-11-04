# FollowMath AI

**Accessible math for students who cannot see the whiteboard.**

FollowMath AI helps a blind or visually impaired student follow an equation written on a whiteboard, write it out themselves, and check their own work. Everything is done through sound and touch. The student never needs to see the screen.

Standard educational software inherently assumes the user can see the screen. To solve this, FollowMath AI was engineered entirely from scratch for the Meta AI Global Hackathon. Powered exclusively by Meta's ecosystem (SAM 3.1, Muse Spark, and Muse Transcribe), it features a custom real-time backend and a strictly blind-centric, dual-device architecture. Every single UI choice and WebSocket sync was deliberately built from the ground up to serve the unique tactile and auditory needs of visually impaired students.

---
## The Inspiration

This project originated from a Computer Engineering professor's observations of a visually impaired student in an advanced Master's course. While the student could comprehend the spoken lectures, the inability to spatially interact with, draw, and solve complex math equations created an immense academic barrier, ultimately impacting their performance on written midterms. FollowMath AI was built to solve this exact real-world problem.

## The Problem

Math is taught on boards, slides, and paper. A student who cannot see them has to rely on someone else to read out every symbol, and to check every line they write. But hearing an equation read aloud doesn't convey its 2D structure. Because blind students cannot easily imagine how the math looks without physically drawing it or feeling the spatial layout of the equations, relying on verbal dictation makes it hard to keep up in class and hard to study alone.

## What it does

1. **Reads the board.** A phone is propped up facing the whiteboard. The app takes a picture, finds the equation and reads it aloud.
2. **Keeps up with the instructor.** The phone keeps watching the board. When the instructor writes more, the app updates and tells the student using audio cues.
3. **Guides the student's writing.** The student writes the equation on an iPad with a stylus. The app stays quiet while the student is on track, and says things like "Move right to reach the middle of the equation" when they drift.
4. **Checks the work.** When the student pauses, the app looks at what they wrote and gives short spoken feedback, like a patient tutor.
5. **Lets the student feel their own writing.** After writing, the student slides a finger over the screen. The app names each symbol under the finger, such as "the letter y", "an equals sign", "the digit 2". If a symbol is unfinished or messy, it says so honestly (for example "a single horizontal line" or "a scribble") instead of guessing what the student meant.
6. **Answers spoken questions.** The microphone button is permanently anchored in the extreme bottom-right corner of the screen. Because a visually impaired user cannot see floating digital menus, fixing this button against the physical edge allows them to instantly find it by touch without guessing. The student simply taps this tactile corner, asks a question, and taps again to send it.
7. **Blind-First Palm Rejection.** Because a blind student cannot rely on visual cues to find the edge of the screen, they must use their hands to navigate the physical space. The canvas strictly ignores hand contact, allowing the blind user to safely rest their non-dominant hand flat on the screen as a tactile ruler and spatial anchor without ever drawing accidental marks.

---

## How it works

Two devices connect to one small server on your computer.

```
   Phone (camera)                 Your computer                 iPad (student)
   /camera page        <-->       FastAPI server       <-->     main page "/"
   watches the board              runs the AI calls             draws, touches, listens
```

| Step | What happens | Model or tool |
|------|--------------|---------------|
| Find the equation on the board | The photo is searched for written symbols | SAM 3.1 |
| Read and describe symbols | Each symbol is cropped and named, and the board and the student's drawing are described and checked | Muse Spark |
| Understand spoken questions | The student's voice is turned into text | Muse Transcribe |
| Speak to the student | Text is turned into speech | gTTS (Google text-to-speech) |

The code that finds and names the symbols the student can touch is in `symbol_map.py`. Everything else is in `big_bang.py`.

---

## What you need

- A computer to run the server (Python 3.10 or newer)
- A phone with a back camera
- A tablet with a stylus (Apple Pencil or similar)
- A Meta API key with access to the models listed above
- [ffmpeg](https://ffmpeg.org/download.html) installed and available on your PATH (used to convert the student's voice recording)
- An internet connection (the AI models and the speech voice run online)

---

## Setup

**1. Get the code and install the packages**

```bash
git clone <your-repository-url>
cd FollowMathAI
python -m venv venv
```

Activate the virtual environment:

- Windows: `venv\Scripts\activate`
- Mac or Linux: `source venv/bin/activate`

Then install the packages:

```bash
pip install -r requirements.txt
```

**2. Add your API key**

Copy `.env.example` to a new file named `.env` and put your key in it:

```
META_API_KEY=your_key_here
```

Never share your `.env` file or upload it to GitHub.

**3. Check that ffmpeg works**

```bash
ffmpeg -version
```

If you see a version number, you are ready. If not, install ffmpeg and try again.

---

## Run it

Start the server:

```bash
uvicorn big_bang:app --host 0.0.0.0 --port 8000
```

You will see output in the terminal as the app receives actions from the devices. This is normal.

### Connect your devices over HTTPS

Phone and iPad browsers only allow the camera and microphone on secure (HTTPS) pages. A plain `http://` address on your network will not work. The easiest fix is a tunnel service that gives your local server a secure link, for example:

```bash
ngrok http 8000
```

Use the `https://...` address it prints.

- **iPad:** open that address.
- **Phone:** open the same address followed by `/camera`.

---

## How to use it

1. **Phone (Set up by a peer or instructor):** open the `/camera` page, tap anywhere to start the camera, then prop the phone up so it faces the whiteboard. Allow camera access when asked.
2. **iPad (Operated by the blind student):** Open the main page. A screen reader (like VoiceOver) or a single tap anywhere on the screen will start the app with a spoken welcome.
3. Wait for the instructions while the app reads the equation aloud.
4. **Write with the stylus.** Listen for guidance to help complete writing if missing or if you drift off track.
5. **Pause.** The app checks what you wrote and tells you how it looks.
6. **Explore with a finger.** Slide a finger over your writing to hear each symbol named. Fingers explore, and the stylus draws.
7. **Ask a question.** Tap the microphone button (bottom right), speak, then tap it again to send.

### Voice commands

| Say something like | What happens |
|--------------------|--------------|
| "Could you undo the last stroke?" | Removes the last stroke |
| "Could you clear the canvas" | Clears the canvas |
| Anything else, such as "Am I on the right track?" | Sent to the AI tutor, along with your current drawing |

---

## Project files

| File | What it does |
|------|--------------|
| `big_bang.py` | The server and the web pages for the iPad and phone: scanning, guidance, tutor feedback, voice, live updates |
| `symbol_map.py` | Finds each symbol in the student's writing and names it for touch exploration |
| `requirements.txt` | Python packages to install |
| `.env.example` | Template for your API key file |

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| "The phone camera is not connected" | Open the `/camera` page on the phone first, then tap the iPad again |
| Camera or microphone does not start | Make sure you opened the secure `https://` address and allowed access in the browser |
| No spoken voice | Tap the iPad screen once after loading (browsers block sound until you tap), and check the volume |
| Voice questions are not understood | Check that `ffmpeg -version` works and that your API key is correct |
| Symbols are not named | Check your API key. Without it, the app falls back to simple shape detection and names are less accurate |
| Scan does not read the equation | Use good light, hold the phone still, and write the equation large and clear |

---

## Limits to know about

- The app is optimized for visually impaired individuals with prior non-Braille writing experience (e.g., those who recently lost their vision and retain muscle memory for standard handwriting).
- The app serves **one student at a time**. Two iPads connected to the same server will interfere with each other.
- Speech and AI features **need internet**.
- Spoken output is **English only**.
- Pictures of the board and the student's writing are sent to Meta's API to be processed.

---

## Credits

- [SAM 3.1](https://ai.meta.com), Muse Spark and Muse Transcribe by Meta
- [gTTS](https://pypi.org/project/gTTS/) for text-to-speech
- [FastAPI](https://fastapi.tiangolo.com/), [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/) and [httpx](https://www.python-httpx.org/)

Built by **[Aafia Nawal / FollowMathAI]**.
