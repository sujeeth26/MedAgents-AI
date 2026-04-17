import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import os

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports.db")

def init_db():
    """Initialize the database and create tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        chief_complaint TEXT,
        summary TEXT,
        symptoms TEXT,
        medications TEXT,
        advice TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")

def save_report(report_data: Dict) -> str:
    """Save a new report to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    report_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Ensure complex types are JSON serialized
    symptoms = json.dumps(report_data.get('symptoms', []))
    medications = json.dumps(report_data.get('medications', []))
    advice = json.dumps(report_data.get('advice', []))
    
    cursor.execute('''
    INSERT INTO reports (id, timestamp, chief_complaint, summary, symptoms, medications, advice)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        report_id,
        timestamp,
        report_data.get('chief_complaint', ''),
        report_data.get('summary', ''),
        symptoms,
        medications,
        advice
    ))
    
    conn.commit()
    conn.close()
    return report_id

def get_reports() -> List[Dict]:
    """Retrieve all reports from the database, ordered by timestamp desc."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM reports ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    reports = []
    for row in rows:
        reports.append({
            'id': row['id'],
            'timestamp': row['timestamp'],
            'chief_complaint': row['chief_complaint'],
            'summary': row['summary'],
            'symptoms': json.loads(row['symptoms']),
            'medications': json.loads(row['medications']),
            'advice': json.loads(row['advice'])
        })
    
    conn.close()
    return reports

def get_report(report_id: str) -> Optional[Dict]:
    """Retrieve a specific report by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    row = cursor.fetchone()
    
    if row:
        report = {
            'id': row['id'],
            'timestamp': row['timestamp'],
            'chief_complaint': row['chief_complaint'],
            'summary': row['summary'],
            'symptoms': json.loads(row['symptoms']),
            'medications': json.loads(row['medications']),
            'advice': json.loads(row['advice'])
        }
        conn.close()
        return report
    
    conn.close()
    return None
