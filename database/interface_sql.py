import array
import sqlite3 as sqlite
import os
import sys
import numpy as np

path = os.path.dirname(__file__) + "/database.db"
con = sqlite.connect(path)


def create_database(name_table):
    """
    function: Create Table contain Information of FaceID
    with: Id, FullName, Face, Embed
    Args:
        name_table: name_table
    Returns:
    """
    with con:
        cur = con.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {name_table}")
        cur.execute(f"CREATE TABLE {name_table}(id TEXT PRIMARY KEY, fullname TEXT, face BLOB, embed BLOB)")
        cur.close()


def create_database1(name_table):
    """
    function: Create Table contain Information of Action
    with: Id, FullName, Action, Time
    Args:
        name_table: name_table
    Returns:
    """
    with con:
        cur = con.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {name_table}")
        cur.execute(f"CREATE TABLE {name_table}(id TEXT, fullname TEXT, face BLOB, action TEXT, time TEXT)")
        cur.close()


def add_face(data_tuple, name_table='faceid'):
    """
    Function: add new face
    Args:
        data_tuple: tuple
        name_table:name of table
    Returns:
    """

    try:
        con = sqlite.connect(path)
        cursor = con.cursor()
        print("Connected to SQLite")
        sqlite_insert_blob_query = f""" INSERT INTO {name_table}(id, fullname, face, embed) VALUES (?, ?, ?, ?)"""
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        con.commit()
        print("Data inserted successfully into a table")
        cursor.close()

    except sqlite.Error as error:
        print("Failed to insert data into sqlite table", error)
    finally:
        if con:
            con.close()
            print("the sqlite connection is closed")


def get_all_face(name_table='faceid'):
    """
    Function: Get information of name_table
    Args:
        name_table: name of table
    Returns:
    """
    con = sqlite.connect(path)
    with con:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM {name_table}")
        # get data
        rows = cur.fetchall()
        id_face, fullname, face, embed = [], [], [], []
        for id, row in enumerate(rows):
            row = list(row)
            # convert binary to array
            arr = np.frombuffer(row[len(row)-1], dtype='float')
            # convert array 1D to nD
            v_emb = arr.reshape(len(arr)//512, 512).ravel().tolist()
            # convert image
            image = np.frombuffer(row[len(row)-2], dtype='uint8')
            image = image.reshape(112, 112, 3)
            id_face.append(row[0])
            fullname.append(row[1])
            face.append(image)
            embed.append(v_emb)
        return id_face, fullname, face, embed


def get_all_action(name_table='action_data'):
    """
    Function: Get information of name_table
    Args:
        name_table: name of table
    Returns:
    """
    con = sqlite.connect(path)
    with con:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM {name_table}")
        # get data
        rows = cur.fetchall()
        id_face, fullname, face, action, time_ac = [], [], [], [], []
        for id, row in enumerate(rows):
            row = list(row)
            # convert image
            image = np.frombuffer(row[len(row)-3], dtype='uint8')
            image = image.reshape(112, 112, 3)
            id_face.append(row[0])
            fullname.append(row[1])
            face.append(image)
            action.append(row[3])
            time_ac.append(row[4])
        return id_face, fullname, face, action, time_ac


def update_info(data_change, name_table='faceid'):
    """
    Function: Again update information of face
    Args:
        name_rows: name of change rows
        value_replace: change value or name
        id: id of employee
        full_name: full name of face
        name_table: name of table
    Returns:
    """
    try:
        con = sqlite.connect(path)
        cursor = con.cursor()
        print("Connected to SQLite")
        sql_update_query = f"""UPDATE {name_table} SET fullname = ? Where id = ?"""
        cursor.execute(sql_update_query, (data_change[1], data_change[0]))
        con.commit()
        print("Record update successfully")
        cursor.close()

    except sqlite.Error as error:
        print("Failed to update record from a sqlite table", error)
    finally:
        if con:
            con.close()
            print("sqlite connection is closed")


def delete_face(code_id: str, name_table='faceid'):
    """
    function: Remove a object with id and full_name at name_table
    Args:
        id: data id
        name_table: name of table
    Returns:
    """
    try:
        con = sqlite.connect(path)
        cursor = con.cursor()
        print("Connected to SQLite")
        sql_update_query = f"""DELETE FROM {name_table} Where id = ?"""
        cursor.execute(sql_update_query, (code_id, ))
        con.commit()
        print("Record deleted successfully")
        cursor.close()

    except sqlite.Error as error:
        print("Failed to delete record from a sqlite table", error)
    finally:
        if con:
            con.close()
            print("sqlite connection is closed")


def delete_all_task(name_table='action_data'):
    try:
        sql = f"""DELETE FROM {name_table}"""
        con = sqlite.connect(path)
        cursor = con.cursor()
        cursor.execute(sql)
        con.commit()
        print('Delete successfully.')
        cursor.close()
    except sqlite.Error as error:
        print('Failed to delete from a sqlite table', error)
    finally:
        if con:
            con.close()
            print("sqlite connection is closed")


def add_action(data_tuple, name_table='action_data'):
    """
    Function: add new face
    Args:
        data_tuple: tuple
        name_table:name of table
    Returns:
    """

    try:
        con = sqlite.connect(path)
        cursor = con.cursor()
        # print("Connected to SQLite")
        sqlite_insert_blob_query = f""" INSERT INTO {name_table}
        (id, fullname, face, action, time) VALUES (?, ?, ?, ?, ?)"""
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        con.commit()
        # print("Data inserted successfully into a table")
        cursor.close()

    except sqlite.Error as error:
        print("Failed to insert data into sqlite table", error)
    finally:
        if con:
            con.close()
            print("the sqlite connection is closed")


if __name__ == '__main__':
    # create database for tab employee
    # create_database('faceid')
    # create_database1('action_data')
    # # add info
    # embed = []
    # a = np.ones((1, 512))[0].tolist()
    # b = np.zeros((1, 512))[0].tolist()
    # embed = np.array(b, dtype='float')
    # face = np.ones((112, 112, 3), dtype='uint8')
    # add_face(('DEV01', 'Dao Duy Ngu', face, embed), 'faceid')
    # add_employee(('DEV02', 'Nguyen Vu Hoai Duy', 1, 'Dev', 'AI', embed), 'employee')
    # add_employee(('DEV03', 'Tran Chi Cuong', 0, 'Dev', 'AI', embed), 'employee')
    # # delete employee
    # delete_employee('DEV01')
    # # get employee
    # data = get_all_face('faceid')
    # print(data)
    # print(data)
    # # change info
    # update_info(('DEV01', 'Dao Duy Tu'))
    delete_all_task()
    # data = get_all_employee('employee')
    # print(data)
    # insert_timekeeping('DEV02', 'Dao Duy Ngu')