import psycopg2 as pg

data_connect = {
    "host": "192.168.1.56",
    "port": "5432",
    "user": "admin",
    "password": "admin2222222",
    "dbname": "taskdns",
}  # данные для подключения к БД taskdns


try:
    connection = pg.connect(**data_connect)

    connection.autocommit = True

    # with connection.cursor() as cursor:
    #     cursor.execute("CREATE TABLE t_cities()")

    #     print("Копирование выполнено успешно")

    # with connection.cursor() as cursor, open(
    #     "C:/Users/Kirill/Documents/relocatetodb/t_cities.csv", "r", encoding="utf-8"
    # ) as file:
    #     file.readline()
    #     for row in file:
    #         cursor.execute(
    #             "INSERT INTO t_cities(stolbec1, stolbec2, stolbec3) VALUES(%s, %s, %s);",
    #             row.strip().split(","),
    #         )

    #     print("Копирование выполнено успешно")

    with connection.cursor() as cursor, open(
        "C:/Users/Kirill/Documents/relocatetodb/t_cities.csv", "r", encoding="utf-8"
    ) as file:
        file.readline()
        cursor.copy_from(file, "t_cities", sep=",")


except Exception as ex:
    print("[INFO] Error while working with PostgreSQL", ex)

finally:
    if connection:
        connection.close()
        print("Connection to SQL close")
