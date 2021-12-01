#!/usr/bin/env python3

import hashlib
import os
import sqlite3
from img2dataset.config import DB_FILE


def hash_image(data):
    """
    Create a return a hash from an image

    Returns:
        Hash of the picture
    """

    
    m = hashlib.md5()
    while True:
        if not data:
            break
        m.update(data)
    return m.hexdigest()


def add_to_db(img_hash, thread_nb):
    """
    Add a thread number to Image_Hash table
    """

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("INSERT INTO Image_Hash (hash, Thread_Number) VALUES (?,?)", (img_hash, thread_nb))

    conn.commit()
    conn.close()


def is_duplicate(img_hash):
    """
    Check if a picture with the same img_hash was already downloaded. (Since img2dataset's DB creation)

    Returns:
        True if the picture was already downloaded before, False otherwise
    """

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT Hash FROM Image_Hash WHERE Hash = ?", (img_hash,))
    result = c.fetchone()

    conn.close()

    if result:
        return True
    else:
        return False