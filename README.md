# Automated Caesar Cipher Decoder

This project is a Flask-based web application that automatically decodes Caesar Cipher-encrypted text using frequency analysis and a machine learning model. It intelligently predicts the shift key using a trained Random Forest Classifier and provides decoded output.

---

## Features

- **Frequency Analysis** for initial decryption guesses.
- **Machine Learning Model** (Random Forest) trained on Caesar cipher patterns.
- **Web Interface** built using Flask, HTML, CSS, and JavaScript.
- **Accurate Predictions** based on letter distribution in ciphertext.

## Project Structure

├── app.py                  # Flask app  
├── train_model.py          # Script to train and save the ML model  
├── templates/  
│   └── index.html          # HTML frontend  
├── static/  
│   └── script.js           # JavaScript for async form handling  
├── cipher2.csv             # Dataset for training  
├── allCharFreqs.csv        # Character frequency data   
└── caesar_cipher_model.pkl # Saved ML model (after training)  
