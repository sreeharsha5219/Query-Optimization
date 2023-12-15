import configparser


config = configparser.ConfigParser()
config.read("lero.conf")
config = config["lero"]

PORT = int(config.get("PORT", "5432"))
HOST = config.get("HOST", "localhost")
USER = config.get("USER", "postgres")
PASSWORD = config.get("PASSWORD", "postgres")

DB = str(config["DB"])

CONNECTION_STR = "dbname=" + DB + " user=" + USER + " password=" + PASSWORD + " host=localhost port=" + str(PORT)

TIMEOUT = int(config["TIMEOUT"])
PG_DB_PATH = str(config["PG_DB_PATH"])

LERO_SERVER_PORT = int(config.get("LERO_SERVER_PORT", "14567"))
LERO_SERVER_HOST = config.get("LERO_SERVER_HOST", "localhost")
LERO_SERVER_PATH = str(config["LERO_SERVER_PATH"])
LERO_DUMP_CARD_FILE = "dump_card_with_score.txt"

LOG_PATH = str(config["LOG_PATH"])
SEP = str(config.get("SEP", "#####"))
