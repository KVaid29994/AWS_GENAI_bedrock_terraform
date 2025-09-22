from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

def get_conversation_chain():
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region="us-east-1"
    )
    
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""You are a helpful assistant.
                Conversation so far:
                {history}
                Human: {input}
                Assistant:"""

    )
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )

    return chain
