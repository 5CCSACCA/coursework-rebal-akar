# tests/test_users.py

import pytest
from datetime import datetime, timedelta
from jose import jwt
from core.config import settings
from core.security import get_password_hash, verify_password
from database.mongodb import mongodb

@pytest.mark.anyio
class TestUserRegistration:

    async def test_register_user_success(self, test_client, new_user_data):
        """
        Test successful user registration.
        """
        response = await test_client.post("/users/register", json=new_user_data)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == new_user_data["username"]
        assert data["email"] == new_user_data["email"]
        assert "id" in data

        # Verify that the user is stored in the database with hashed password
        user_in_db = await mongodb.db.users.find_one({"username": new_user_data["username"]})
        assert user_in_db is not None
        assert user_in_db["email"] == new_user_data["email"]
        assert verify_password(new_user_data["password"], user_in_db["hashed_password"]) == True

    async def test_register_user_existing_username(self, test_client, new_user_data):
        """
        Test registration with an existing username.
        """
        # First, register the user
        await test_client.post("/users/register", json=new_user_data)

        # Attempt to register again with the same username
        new_user_data["email"] = "newemail@example.com"
        response = await test_client.post("/users/register", json=new_user_data)
        assert response.status_code == 400
        assert response.json()["detail"] == "Username or email already registered"

    async def test_register_user_existing_email(self, test_client, new_user_data):
        """
        Test registration with an existing email.
        """
        # First, register the user
        await test_client.post("/users/register", json=new_user_data)

        # Attempt to register again with the same email
        new_user_data["username"] = "newusername"
        response = await test_client.post("/users/register", json=new_user_data)
        assert response.status_code == 400
        assert response.json()["detail"] == "Username or email already registered"

    async def test_register_user_invalid_email(self, test_client, new_user_data):
        """
        Test registration with an invalid email format.
        """
        new_user_data["email"] = "invalid-email"
        response = await test_client.post("/users/register", json=new_user_data)
        assert response.status_code == 422  # Pydantic validation error
        assert any(error["loc"][-1] == "email" for error in response.json()["detail"])

    @pytest.mark.parametrize("password, error_msg", [
        ("weakpass", "Password must be at least 8 characters long"),
        ("Short1!", "Password must be at least 8 characters long"),
        ("noUpper1!", "Password must contain at least one uppercase letter"),
        ("NOLOWER1!", "Password must contain at least one lowercase letter"),
        ("NoDigit!", "Password must contain at least one digit"),
        ("NoSpecial1", "Password must contain at least one special character"),
    ])
    async def test_register_user_weak_password(self, test_client, new_user_data, password, error_msg):
        """
        Test registration with passwords that do not meet complexity requirements.
        """
        new_user_data["password"] = password
        response = await test_client.post("/users/register", json=new_user_data)
        assert response.status_code == 422  # Pydantic validation error
        assert any(error["msg"] == error_msg for error in response.json()["detail"])

    async def test_register_user_empty_username(self, test_client, new_user_data):
        """
        Test registration with an empty username.
        """
        new_user_data["username"] = ""
        response = await test_client.post("/users/register", json=new_user_data)
        assert response.status_code == 422

    async def test_register_user_empty_email(self, test_client, new_user_data):
        """
        Test registration with an empty email.
        """
        new_user_data["email"] = ""
        response = await test_client.post("/users/register", json=new_user_data)
        assert response.status_code == 422

    async def test_register_user_empty_password(self, test_client, new_user_data):
        """
        Test registration with an empty password.
        """
        new_user_data["password"] = ""
        response = await test_client.post("/users/register", json=new_user_data)
        assert response.status_code == 422

    async def test_register_user_special_characters_in_username(self, test_client, new_user_data):
        """
        Test registration with special characters in the username.
        """
        new_user_data["username"] = "user!@#"
        response = await test_client.post("/users/register", json=new_user_data)
        # Decide if special characters are allowed in username
        # Assuming they are allowed
        assert response.status_code == 200

@pytest.mark.anyio
class TestUserLogin:

    async def test_login_user_success(self, test_client, new_user_data):
        """
        Test successful user login.
        """
        # Register the user first
        await test_client.post("/users/register", json=new_user_data)

        response = await test_client.post("/users/login", data={
            "username": new_user_data["username"],
            "password": new_user_data["password"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

        # Decode the token and check the payload
        decoded_token = jwt.decode(data["access_token"], settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert decoded_token["sub"] == new_user_data["username"]

    async def test_login_user_incorrect_password(self, test_client, new_user_data):
        """
        Test login with an incorrect password.
        """
        # Register the user first
        await test_client.post("/users/register", json=new_user_data)

        response = await test_client.post("/users/login", data={
            "username": new_user_data["username"],
            "password": "WrongPassword1!"
        })
        assert response.status_code == 400
        assert response.json()["detail"] == "Incorrect username or password"

    async def test_login_user_nonexistent_user(self, test_client):
        """
        Test login with a non-existent user.
        """
        response = await test_client.post("/users/login", data={
            "username": "nonexistentuser",
            "password": "SomePassword1!"
        })
        assert response.status_code == 400
        assert response.json()["detail"] == "Incorrect username or password"

    async def test_login_user_account_lockout(self, test_client, new_user_data):
        """
        Test account lockout after multiple failed login attempts.
        """
        # Register the user first
        await test_client.post("/users/register", json=new_user_data)

        # Simulate failed login attempts
        for _ in range(10):
            response = await test_client.post("/users/login", data={
                "username": new_user_data["username"],
                "password": "WrongPassword1!"
            })
            assert response.status_code == 400
            assert response.json()["detail"] == "Incorrect username or password"

        # The account should now be locked
        response = await test_client.post("/users/login", data={
            "username": new_user_data["username"],
            "password": new_user_data["password"]
        })
        assert response.status_code == 403
        assert "Account locked until" in response.json()["detail"]

        # Check that account_locked_until is set in the database
        user_in_db = await mongodb.db.users.find_one({"username": new_user_data["username"]})
        assert user_in_db["account_locked_until"] > datetime.utcnow()

    async def test_failed_login_attempts_reset_after_successful_login(self, test_client, new_user_data):
        """
        Test that failed login attempts are reset after a successful login.
        """
        # Register the user first
        await test_client.post("/users/register", json=new_user_data)

        # Simulate failed login attempts
        for _ in range(5):
            response = await test_client.post("/users/login", data={
                "username": new_user_data["username"],
                "password": "WrongPassword1!"
            })
            assert response.status_code == 400

        # Check that failed_login_attempts is 5
        user = await mongodb.db.users.find_one({"username": new_user_data["username"]})
        assert user["failed_login_attempts"] == 5

        # Successful login
        response = await test_client.post("/users/login", data={
            "username": new_user_data["username"],
            "password": new_user_data["password"]
        })
        assert response.status_code == 200

        # Check that failed_login_attempts is reset to 0
        user = await mongodb.db.users.find_one({"username": new_user_data["username"]})
        assert user["failed_login_attempts"] == 0

    async def test_login_user_locked_out_user(self, test_client, new_user_data):
        """
        Test login with an account that is currently locked.
        """
        # Register the user first
        await test_client.post("/users/register", json=new_user_data)

        # Lock the account manually for testing
        await mongodb.db.users.update_one(
            {"username": new_user_data["username"]},
            {"$set": {"account_locked_until": datetime.utcnow() + timedelta(minutes=15)}}
        )

        response = await test_client.post("/users/login", data={
            "username": new_user_data["username"],
            "password": new_user_data["password"]
        })
        assert response.status_code == 403
        assert "Account locked until" in response.json()["detail"]

@pytest.mark.anyio
class TestToken:

    async def test_token_creation_and_decoding(self, test_client, new_user_data):
        """
        Test JWT token creation and decoding.
        """
        # Create a token
        data = {"sub": new_user_data["username"]}
        token = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

        # Decode the token
        decoded_data = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert decoded_data["sub"] == new_user_data["username"]

    # To test token expiration, we need to mock datetime.utcnow()
    # This requires additional setup or external libraries like freezegun
    # For now, we can assume the token expires correctly

class TestSecurity:

    def test_password_hashing(self):
        """
        Test password hashing and verification.
        """
        password = "Test@1234"
        hashed_password = get_password_hash(password)
        assert hashed_password != password  # Ensure password is hashed
        assert verify_password(password, hashed_password) == True
        assert verify_password("WrongPass@123", hashed_password) == False
