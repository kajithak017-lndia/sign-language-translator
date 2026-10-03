# Sign Language Translator

A camera-based sign language translator that uses MediaPipe Holistic landmarks and a TensorFlow CNN-BiLSTM classifier to recognize a small trained vocabulary and display the result in a browser.

## 🚀 Features

- Real-time sign recognition for `hello`, `no`, `please`, `sorry`, `thank_you`, and `yes`
- Rolling 30-frame prediction window with stable-result filtering
- Confidence scores and translation history
- Optional browser text-to-speech output
- Spring Boot API gateway with a Python Flask recognition service
- Health checks at `/api/health` and `/health`

## 🛠️ Technologies Used

- Java
- Spring Boot
- Python
- TensorFlow
- OpenCV
- MediaPipe
- HTML
- CSS
- JavaScript
- Maven
- Visual Studio Code (VS Code)

## Project Structure

```
signlanguage/
├── sign-language-translator/
│   ├── backend-python/app.py
│   ├── data-collection/
│   ├── model-training/
│   ├── data/raw/
│   ├── models/
│   └── preprocessing.py
└── sign-translator-backend/
    └── src/main/resources/static/index.html
```

## How to Run

Open two VS Code terminals.

Terminal 1, start the Python recognition service:

```powershell
cd D:\signlanguage\sign-language-translator
.\venv\Scripts\Activate.ps1
python .\backend-python\app.py
```

Terminal 2, start the Spring Boot web application:

```powershell
cd D:\signlanguage\sign-translator-backend
./mvnw.cmd spring-boot:run
```

Open `http://localhost:8080`. The Python service listens on port 5000 and the browser-facing Spring Boot service listens on port 8080.

## 🎯 Future Enhancements

- Support for additional sign languages
- Larger and more diverse training data
- Mobile application support

## 👩‍💻 Author

**Kajitha K**

Computer Science and Engineering Student

## 📄 License

This project is created for educational and learning purposes.
