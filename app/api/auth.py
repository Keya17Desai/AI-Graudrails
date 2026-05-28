from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.auth.users import authenticate
from app.auth.jwt import create_access_token
from app.auth.dependencies import get_current_user
from app.monitoring.metrics import auth_attempts_total

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    full_name: str


class MeResponse(BaseModel):
    username: str
    role: str
    full_name: str


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    user = authenticate(request.username, request.password)
    if not user:
        auth_attempts_total.labels(result="failure").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    auth_attempts_total.labels(result="success").inc()
    token = create_access_token({"sub": request.username})
    return TokenResponse(
        access_token=token,
        role=user["role"],
        full_name=user["full_name"],
    )


@router.get("/me", response_model=MeResponse)
def me(current_user: dict = Depends(get_current_user)):
    return MeResponse(**current_user)
