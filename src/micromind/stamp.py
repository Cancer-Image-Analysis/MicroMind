from datetime import datetime
from uuid import uuid4


def timestamp():
    return datetime.now()


def uuid():
    return str(uuid4())


def eventid():
    return f"{timestamp()}-{uuid()}".replace(" ", "-")
