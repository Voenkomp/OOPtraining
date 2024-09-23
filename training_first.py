import psycopg2 as pg
import time

data_connect = {
    "host": "192.168.1.56",
    "port": "5432",
    "user": "admin",
    "password": "admin2222222",
    "dbname": "taskdns",
}  # данные для подключения к БД taskdns

connection = pg.connect(**data_connect)

try:
    begin = time.time()
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, branch FROM t_branches")
        result = cursor.fetchall()
        # print(result)
        for id in result:
            cursor.execute(
                """INSERT INTO result17
                    SELECT product, SUM(count) AS total
                FROM t_sales
                WHERE branch = %s
                GROUP BY product
                ORDER BY total DESC
                LIMIT 10
            """,
                [id[0]],
            )

    print(f"запрос выполнен за {time.time() - begin}")

    # with connection.cursor() as cursor:
    #     data = [(1, "Алиса"), (2, "Боб"), (3, "Чарли")]
    #     args_str = ",".join(cursor.mogrify("(%s,%s)", x).decode("utf-8") for x in data)
    #     cursor.execute("INSERT INTO таблица (колонка1, колонка2) VALUES " + args_str)
    #     connection.commit()

    # with connection.cursor() as cursor:
    #     cursor.execute("SELECT * FROM t_cities LIMIT 10")
    #     print(cursor.fetchmany(10))

except Exception as ex:
    print("[INFO] Error while working with PostgreSQL", ex)
finally:
    if connection:
        connection.close()
        print("Connection to PostgreSQl close")
