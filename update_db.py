import asyncio
import logging

from database import PostgreSQL
import config
from crud import auth_1c
from pyschemas import Employee1C

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_users_from_1c():
    """
    Обновляет ФИО и должность всех пользователей в Postgres по данным из 1С.
    """
    postgres = PostgreSQL(**config.postgres_config)
    try:
        postgres.cursor.execute("SELECT user_id FROM users")
        user_ids = [row[0] for row in postgres.cursor.fetchall()]

        for user_id in user_ids:
            try:
                employee = asyncio.run(auth_1c(user_id))
                if employee and isinstance(employee, Employee1C):
                    firstname = employee.fio.split()[1]
                    lastname = employee.fio.split()[0]
                    postgres.cursor.execute(
                        "UPDATE users SET first_name = %s, last_name = %s WHERE user_id = %s",
                        (firstname, lastname, user_id)
                    )
                    postgres.connection.commit()
                    logger.info(f"Пользователь {user_id} обновлён: {firstname}, {lastname}")
            except Exception as e:
                logger.warning(f"Ошибка при обновлении пользователя {user_id}: {e}")
                postgres.connection.rollback()
    finally:
        postgres.connection_close()

if __name__ == "__main__":
    update_users_from_1c()