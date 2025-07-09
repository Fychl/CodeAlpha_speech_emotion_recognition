📦 Speech Emotion Recognition (SER)
Code Alpha Internship – Task 2

This project aims to detect human emotions (such as happy, sad, angry, etc.) from speech signals using deep learning techniques.
It uses popular datasets and a pipeline built with Python and libraries like Librosa, TensorFlow/Keras, and Scikit-learn.

📂 Datasets Used
✅ TESS (Toronto emotional speech set)

Contains: 2800+ audio clips by 2 actresses, labeled with 7 emotions

Download: TESS Dataset

✅ SAVEE (Surrey Audio-Visual Expressed Emotion)

Contains: ~480 utterances from 4 male actors, labeled with 7 emotions

Download: SAVEE Dataset

✅ CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

Contains: ~7,400 clips from 91 actors, labeled with 6 basic emotions + neutral

Download: CREMA-D Dataset

⚠️ Note: These datasets are free for research; check their licenses for any use beyond academic / research.

🛠️ Project Structure
bash
Copy
Edit
.
├── data/                  # Datasets (downloaded & extracted)
├── features/              # Extracted MFCC / spectrogram features
├── models/                # Saved models (H5 / PB files)
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── Main.py                   # Python source code 
├── requirements.txt
└── README.md
📊 Features
Feature extraction: MFCCs (Mel-frequency cepstral coefficients)

Deep learning models: CNN, LSTM, RNN

Visualization: Confusion matrix, loss & accuracy curves

Evaluation: Accuracy, precision, recall, F1 score

🚀 How to Run
1️⃣ Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
2️⃣ Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Download datasets & extract into data/ folder.


bash
Copy
Edit
python src/Main.py
6️⃣ Evaluate & test:


✏️ Contribution & Credits
Built by: Utkarsh Tiwari 

Internship: Code Alpha (Task 2)

Special thanks to datasets authors & open source community.

📄 License
For academic and research use only.
Check dataset-specific licenses before commercial use.
