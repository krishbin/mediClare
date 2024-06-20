import sqlite3
import os
import uuid

class DatabaseService:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.initialize_tables()

    def __del__(self):
        self.conn.close()
    
    def create_input_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS input (
                uuid TEXT PRIMARY KEY,
                prompt TEXT
            )
            """
        )
    
    def initialize_tables(self):
        if not os.path.exists(self.db_path):
            self.create_input_table()
            self.create_output_table()
            self.create_feedback_table()
    
    def create_output_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS output (
                uuid TEXT PRIMARY KEY,
                output TEXT,
                output_type TEXT,
                foreign key(uuid) references users(input)
            )
            """
        )
    
    def generate_uuid(self):
        uuid = str(uuid.uuid4())
        if self.cursor.execute(f"SELECT * FROM input WHERE uuid = {uuid}").fetchone() is not None:
            return self.generate_uuid()
        return uuid
    
    def create_feedback_table(self,uuid):
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS feedback (
                {uuid} TEXT PRIMARY KEY,
                feedback TEXT,
                feedback_type TEXT,
                foreign key(uuid) references users(output)
            )
            """
        )
    
    def insert_input(self, prompt):
        uuid = self.generate_uuid()
        self.cursor.execute(
            f"""
            INSERT INTO input (
                {uuid},
                {prompt}
                )
                """
        )
    
    def insert_output(self, uuid, output, output_type):
        self.cursor.execute(
            f"""
            INSERT INTO output (
                {uuid},
                {output},
                {output_type}
                )
                """
        )
    
    def insert_feedback(self, uuid, feedback, feedback_type):
        self.cursor.execute(
            f"""
            INSERT INTO feedback (
                {uuid},
                {feedback},
                {feedback_type}
                )
                """
        )

    def get_output(self, uuid):
        self.cursor.execute(
            f"""
            SELECT output
            FROM output
            WHERE uuid = {uuid}
            """
        )
        return self.cursor.fetchone()
    
    def get_input(self, uuid):
        self.cursor.execute(
            f"""
            SELECT prompt
            FROM input
            WHERE uuid = {uuid}
            """
        )
        return self.cursor.fetchone()

    def get_random_input(self):
        self.cursor.execute(
            """
            SELECT *
            FROM input
            ORDER BY RANDOM()
            LIMIT 1
            """
        )
        return self.cursor.fetchone()

    