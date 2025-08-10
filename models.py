from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class PredictionEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    disease = db.Column(db.String(64), nullable=False)
    prediction = db.Column(db.String(64), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PredictionEntry {self.username} {self.disease} {self.prediction}>'
