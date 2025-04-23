from datetime import datetime

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import JSON, Field, SQLModel, delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

### ORM Models


class Chat(SQLModel, table=True):
    __tablename__ = "Chat"
    id: str = Field(default=None, primary_key=True)
    createdAt: str
    title: str
    userId: str
    visibility: str = "private"


class Message(SQLModel, table=True):
    __tablename__ = "Message"
    id: str = Field(default=None, primary_key=True)
    chatId: str
    role: str
    content: dict = Field(sa_type=JSON)
    createdAt: str


class TeamState(SQLModel, table=True):
    __tablename__ = "TeamState"
    type: str
    chatId: str = Field(default=None, primary_key=True)
    agentStates: dict = Field(sa_type=JSON)


### Utility Functions


def create_db_engine(postgres_url: str) -> AsyncEngine:
    # connect to postgres db
    # Replace postgres:// with postgresql:// for SQLAlchemy
    postgres_url = postgres_url.replace("postgres://", "postgresql+asyncpg://", 1)
    db = create_async_engine(postgres_url)
    return db


async def _ensure_table(session: AsyncSession, table) -> None:
    """
    Create table on first access.
    SQLModel / SQLAlchemy's DDL calls can only be executed in a synchronous context,
    so we need to proxy to the synchronous engine via `run_sync()`.
    """

    def _sync_create(sync_session: AsyncSession):
        # Use the inspector from sqlalchemy to check if table exists
        engine = sync_session.get_bind()
        if not inspect(engine).has_table(table.__tablename__):
            SQLModel.metadata.create_all(engine, tables=[table.__table__])

    await session.run_sync(_sync_create)


### CRUD Functions


async def getChatById(session: AsyncSession, chat_id: str) -> Chat | None:
    await _ensure_table(session, Chat)

    stmt = select(Chat).where(Chat.id == chat_id)
    result = await session.execute(stmt)
    return result.scalars().first()


async def saveChat(
    session: AsyncSession, chat_id: str, user_id: str, title: str
) -> None:
    await _ensure_table(session, Chat)

    chat = Chat(
        id=chat_id,
        createdAt=datetime.utcnow().isoformat(),
        title=title,
        userId=user_id,
    )
    session.add(chat)
    await session.commit()


async def saveMessage(
    session: AsyncSession,
    msg_id: str,
    chat_id: str,
    role: str,
    content: dict,
    created_at: str,
) -> None:
    await _ensure_table(session, Message)

    message = Message(
        id=msg_id,
        chatId=chat_id,
        role=role,
        content=content,
        createdAt=created_at,
    )
    session.add(message)
    await session.commit()


async def deleteChatById(session: AsyncSession, chat_id: str) -> None:
    await _ensure_table(session, Chat)

    # Delete messages first, then delete chat
    await session.exec(delete(Message).where(Message.chatId == chat_id))
    await session.exec(delete(Chat).where(Chat.id == chat_id))
    await session.commit()


async def getStateByChatId(session: AsyncSession, chat_id: str) -> TeamState | None:
    await _ensure_table(session, TeamState)

    stmt = select(TeamState).where(TeamState.chatId == chat_id)
    result = await session.exec(stmt)
    return result.one_or_none()


async def saveTeamState(session: AsyncSession, chat_id: str, state: dict) -> None:
    await _ensure_table(session, TeamState)

    # Check if it exists
    stmt = select(TeamState).where(TeamState.chatId == chat_id)
    result = await session.exec(stmt)
    existing = result.one_or_none()

    if existing:
        existing.agentStates = state["agent_states"]
        session.add(existing)
    else:
        team_state = TeamState(
            chatId=chat_id,
            agentStates=state["agent_states"],
            type=state["type"],
        )
        session.add(team_state)

    await session.commit()
