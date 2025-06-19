from fastapi import APIRouter, Body, HTTPException, Query, status
from starlette.responses import JSONResponse
import logging

import crud
from config import postgres_config
from fastapi import Depends

from database import PostgreSQL
from pyschemas import Search2ResponseData, SearchParams, SearchResponseData
from crud import insert_all_data_from_postgres_to_milvus, insert_wiki_data

router = APIRouter()
logger = logging.getLogger(__name__)



def get_search_params(
    user_id: int = Query(...),
    text: str = Query(...)
) -> SearchParams:
    return SearchParams(user_id=user_id, text=text)

@router.get("/v1/mlv_search", response_model=SearchResponseData, tags=["Milvus"])
async def search_endpoint_with_history(params: SearchParams = Depends(get_search_params)):
    try:
        
        return await crud.search_milvus_and_prep_data(params.text, params.user_id)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/v2/mlv_search", response_model=Search2ResponseData, tags=["Milvus"])
async def search_endpoint(text: str = Query(...)):
    try:
        return await crud.search_milvus(text)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/v1/upload_wiki_data", 
             responses={
                 200: {"description": "Данные успешно загружены"},
                 403: {"description": "Пользователь не является администратором"},
                 500: {"description": "Ошибка сервера при обработке запроса"}
             }, 
             tags=["Milvus"])

async def upload_wiki_data_from_mysqldb_to_milvus(user_data: dict = Body(...)):  
    try:
        user_id = user_data.get("user_id") 
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="user_id is required in request body"
            )
            
        postgres = PostgreSQL(**postgres_config)
        
        # Проверка прав администратора
        if not postgres.check_user_is_admin(user_id):
            logger.warning(f"User {user_id} attempted to upload wiki data without admin rights")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Только администраторы могут загружать данные wiki"
            )

        # Загрузка данных
        wiki_response = insert_wiki_data()
        if not wiki_response:
            logger.error("Failed to insert wiki data")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ошибка при загрузке данных из базы wiki"
            )

        # Перенос данных в Milvus
        milvus_data_count, deleted_data_count = insert_all_data_from_postgres_to_milvus()
        
        # Формирование ответа
        response_message = ""
        if deleted_data_count:
            response_message = f'Обнаружено и удалено {deleted_data_count} дубликатов. '
        response_message += f'Текущее количество записей в базе: {milvus_data_count}'

        logger.info(f"Successfully uploaded wiki data. {response_message}")
        
        return {
            "status": "success",
            "message": response_message,
            "data": {
                "total_records": milvus_data_count,
                "duplicates_removed": deleted_data_count
            }
        }

    except HTTPException:
        raise  # Пробрасываем уже обработанные HTTP исключения
    except Exception as e:
        logger.error(f"Unexpected error during wiki data upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Произошла непредвиденная ошибка при обработке запроса"
        )
    finally:
        # Закрытие соединения с БД
        postgres.connection_close()