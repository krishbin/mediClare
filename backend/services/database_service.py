import mysql.connector as mysql
import uuid
from utils import variables

class DatabaseService:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = mysql.connect(
            host=variables["database_host"],
            user=variables["database_user"],
            password=variables["database_password"],
            database=variables["database_name"]
        )
        self.cursor = self.conn.cursor()
        self.initialize_tables()

    def __del__(self):
        self.conn.close()
    
    def create_input_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS input (
                uuid CHAR(36) PRIMARY KEY,
                prompt TEXT
            )
            """
        )
        self.conn.commit()
    
    def create_output_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS output (
                uuid CHAR(36) PRIMARY KEY,
                output TEXT,
                output_type TEXT,
                input_uuid CHAR(36),
                FOREIGN KEY (input_uuid) REFERENCES input(uuid)
            )
            """
        )
        self.conn.commit()
    
    def create_feedback_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                uuid CHAR(36) PRIMARY KEY,
                feedback TEXT,
                feedback_type TEXT,
                output_uuid CHAR(36),
                FOREIGN KEY (output_uuid) REFERENCES output(uuid)
            )
            """
        )
        self.conn.commit()
    
    def initialize_tables(self):
        self.cursor.execute("SHOW TABLES")
        tables = self.cursor.fetchall()
        if not tables:
            self.create_input_table()
            self.create_output_table()
            self.create_feedback_table()
    
    def generate_uuid(self):
        return str(uuid.uuid4())
    
    def insert_input(self, prompt):
        uuid = self.generate_uuid()
        query = "INSERT INTO input (uuid, prompt) VALUES (%s, %s)"
        self.cursor.execute(query, (uuid, prompt))
        self.conn.commit()
        return uuid
    
    def insert_output(self, input_uuid, output, output_type):
        output_uuid = self.generate_uuid()
        query = "INSERT INTO output (uuid, output, output_type, input_uuid) VALUES (%s, %s, %s, %s)"
        self.cursor.execute(query, (output_uuid, output, output_type, input_uuid))
        self.conn.commit()
        return output_uuid
    
    def insert_feedback(self, output_uuid, feedback, feedback_type):
        feedback_uuid = self.generate_uuid()
        query = "INSERT INTO feedback (uuid, feedback, feedback_type, output_uuid) VALUES (%s, %s, %s, %s)"
        self.cursor.execute(query, (feedback_uuid, feedback, feedback_type, output_uuid))
        self.conn.commit()
        return feedback_uuid

    def get_output(self, output_uuid):
        query = "SELECT output FROM output WHERE uuid = %s"
        self.cursor.execute(query, (output_uuid,))
        return self.cursor.fetchone()
    
    def get_output_from_input(self, input_uuid):
        query = "SELECT * FROM output WHERE input_uuid = %s"
        self.cursor.execute(query, (input_uuid,))
        return self.cursor.fetchone()
    
    def get_input(self, input_uuid):
        query = "SELECT prompt FROM input WHERE uuid = %s"
        self.cursor.execute(query, (input_uuid,))
        return self.cursor.fetchone()

    def get_random_input(self):
        query = "SELECT * FROM input ORDER BY RAND() LIMIT 1"
        self.cursor.execute(query)
        return self.cursor.fetchone()
