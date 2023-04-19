# -*- coding: utf-8 -*-
import faiss
import math
import openai
openai.proxy = {
    "http": "http://192.168.96.7:6666",
    "https": "http://192.168.96.7:6666"
}
import re
import time

from datetime import datetime, timedelta
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseLanguageModel, Document
from langchain.vectorstores import FAISS
from pydantic import BaseModel, Field
from termcolor import colored
from typing import List, Optional, Tuple

# The name you want to use when interviewing the agent.
USER_NAME = "Adam Zhou"
# Can be any LLM you want.
LLM = ChatOpenAI(max_tokens=1500)


class GenerativeAgent(BaseModel):
    """A character with memory and innate characteristics."""
    
    name: Optional[str] = None
    age: Optional[int] = None
    """The traits of the character you wish not to change."""
    traits: Optional[str] = None
    """Current activities of the character."""
    status: Optional[str] = None

    llm: Optional[BaseLanguageModel] = None
    """The retriever to fetch related memories."""
    memory_retriever: Optional[TimeWeightedVectorStoreRetriever] = None
    verbose: bool = False

    """When the total 'importance' of memories exceeds the above threshold, stop to reflect."""
    reflection_threshold: Optional[float] = None

    """The current plan(s) of the agent."""
    current_plan: List[str] = []
    
    summary: Optional[str] = None  #: :meta private:
    summary_refresh_seconds: int = 3600  #: :meta private:
    last_refreshed: datetime = Field(default_factory=datetime.now)  #: :meta private:
    daily_summaries: List[str] = []  #: :meta private:
    memory_importance: float = 0.0  #: :meta private:
    max_tokens_limit: int = 1200  #: :meta private:
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r'\n', text.strip())
        return [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in lines]

    def _compute_agent_summary(self):
        """ Return current summary of the agent."""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{related_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        relevant_memories = self.fetch_memories(f"{self.name}'s core characteristics")
        relevant_memories_str = "\n".join([f"{mem.page_content}" for mem in relevant_memories])
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(name=self.name, related_memories=relevant_memories_str).strip()
    
    def _get_topics_of_reflection(self, last_k: int = 3) -> Tuple[str, str, str]:
        """Return the 3 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            + "Given only the information above, what are the 3 most salient"
            + " high-level questions we can answer about the subjects in the statements?"
            + " Provide each question on a new line.\n\n"
        )
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join([o.page_content for o in observations])
        result = reflection_chain.run(observations=observation_str)
        return self._parse_list(result)
    
    def _get_insights_on_topic(self, topic: str) -> List[str]:
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            "Statements about {topic}\n"
            + "{related_statements}\n\n"
            + "What 5 high-level insights can you infer from the above statements?"
            + " (example format: insight (because of 1, 5, 3))"
        )
        related_memories = self.fetch_memories(topic)
        related_statements = "\n".join([f"{i+1}. {memory.page_content}" for i, memory in enumerate(related_memories)])
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        result = reflection_chain.run(topic=topic, related_statements=related_statements)
        # TODO: Parse the connections between memories and insights
        return self._parse_list(result)
    
    def pause_to_reflect(self) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        print(colored(f"Character {self.name} is reflecting", "blue"))
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(topic)
            for insight in insights:
                self.add_memory(insight)
            new_insights.extend(insights)
        return new_insights
    
    def _score_memory_importance(self, memory_content: str, weight: float = 0.15) -> float:
        """Score the absolute importance of the given memory."""
        # A weight of 0.25 makes this less important than it
        # would be otherwise, relative to salience and time
        prompt = PromptTemplate.from_template("On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: ")
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        score = chain.run(memory_content=memory_content).strip()
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(score[0]) / 10) * weight
        else:
            return 0.0

    def add_memory(self, memory_content: str) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        importance_score = self._score_memory_importance(memory_content)
        self.memory_importance += importance_score
        document = Document(page_content=memory_content, metadata={"importance": importance_score})
        result = self.memory_retriever.add_documents([document])

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (self.reflection_threshold is not None
                and self.memory_importance > self.reflection_threshold
                and self.status != "Reflecting"):
            old_status = self.status
            self.status = "Reflecting"
            self.pause_to_reflect()
            # Hack to clear the importance from reflection
            self.memory_importance = 0.0
            self.status = old_status

        return result
    
    def fetch_memories(self, observation: str) -> List[Document]:
        """Fetch related memories."""
        return self.memory_retriever.get_relevant_documents(observation)
    
    def get_summary(self, force_refresh: bool = False) -> str:
        """Return a descriptive summary of the agent."""
        st = time.time()
        current_time = datetime.now()
        since_refresh = (current_time - self.last_refreshed).seconds
        if not self.summary or since_refresh >= self.summary_refresh_seconds or force_refresh:
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        ed = time.time()
        return f"""
----------------------------------------
Name: {self.name} (age: {self.age})
Innate traits: {self.traits}

Summary:
{self.summary}

Used: {ed - st:.3f}s
----------------------------------------
"""
    
    def get_full_header(self, force_refresh: bool = False) -> str:
        """Return a full header of the agent's status, summary, and current time."""
        summary = self.get_summary(force_refresh=force_refresh)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        return f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"

    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "\nEntity="
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(entity=entity_name, observation=observation).strip()
    
    def _format_memories_to_summarize(self, relevant_memories: List[Document]) -> str:
        content_strs = set()
        content = []
        for mem in relevant_memories:
            if mem.page_content in content_strs:
                continue
            content_strs.add(mem.page_content)
            created_time = mem.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
            content.append(f"- {created_time}: {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])
    
    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        relevant_memories = self.fetch_memories(q1)  # Fetch memories related to the agent's relationship with the entity
        q2 = f"{entity_name} is {entity_action}"
        relevant_memories += self.fetch_memories(q2)  # Fetch things related to the entity-action pair
        context_str = self._format_memories_to_summarize(relevant_memories)
        prompt = PromptTemplate.from_template(
            "{q1}?\nContext from memory:\n{context_str}\nRelevant context: "
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(q1=q1, context_str=context_str.strip()).strip()
    
    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc.page_content)
        return "; ".join(result[::-1])
    
    def _generate_reaction(
        self,
        observation: str,
        suffix: str
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template("{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {recent_observations}"
            + "\nObservation: {observation}"
            + "\n\n" + suffix)
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs = dict(agent_summary_description=agent_summary_description,
                      current_time=current_time_str,
                      relevant_memories=relevant_memories_str,
                      agent_name=self.name,
                      observation=observation,
                      agent_status=self.status)
        consumed_tokens = self.llm.get_num_tokens(prompt.format(recent_observations="", **kwargs))
        kwargs["recent_observations"] = self._get_memories_until_limit(consumed_tokens)
        action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        result = action_prediction_chain.run(**kwargs)
        return result.strip()
    
    def generate_reaction(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            + " what would be an appropriate reaction? Respond in one line."
            + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        self.add_memory(f"{self.name} observed {observation} and reacted by {result}")
        if "REACT:" in result:
            reaction = result.split("REACT:")[-1].strip()
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = result.split("SAY:")[-1].strip()
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            'What would {agent_name} say? To end the conversation, write: GOODBYE: "what to say". Otherwise to continue the conversation, write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        if "GOODBYE:" in result:
            farewell = result.split("GOODBYE:")[-1].strip()
            self.add_memory(f"{self.name} observed {observation} and said {farewell}")
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = result.split("SAY:")[-1].strip()
            self.add_memory(f"{self.name} observed {observation} and said {response_text}")
            return True, f"{self.name} said {response_text}"
        else:
            return False, result


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)


def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    """Runs a conversation between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    print(observation)
    turns = 0
    while True:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(observation)
            print(observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1


if __name__ == "__main__":
    tommie = GenerativeAgent(
        name="Tommie",
        age=25,
        traits="anxious, likes design",  # You can add more persistent traits here.
        status="looking for a job",  # When connected to a virtual world, we can have the characters update their status.
        memory_retriever=create_new_memory_retriever(),
        llm=LLM,
        daily_summaries=["Drove across state to move to a new town but doesn't have a job yet."],
        reflection_threshold=8,  # We will give this a relatively low number to show how reflection works.
    )

    # The current "Summary" of a character can't be made because the agent hasn't made any observations yet.
    print(tommie.get_summary())

    # We can give the character memories directly.
    tommie_memories = [
        "Tommie remembers his dog, Bruno, from when he was a kid",
        "Tommie feels tired from driving so far",
        "Tommie sees the new home",
        "The new neighbors have a cat",
        "The road is noisy at night",
        "Tommie is hungry",
        "Tommie tries to get some rest.",
    ]
    for mem in tommie_memories:
        tommie.add_memory(mem)

    # Now that Tommie has 'memories', their self-summary is more descriptive, though still rudimentary.
    # We will see how this summary updates after more observations to create a more rich description.
    print(tommie.get_summary(force_refresh=True))

    print(interview_agent(tommie, "What do you like to do?"))
    print(interview_agent(tommie, "What are you looking forward to doing today?"))
    print(interview_agent(tommie, "What are you most worried about today?"))

    # Let's have Tommie start going through a day in the life.
    observations = [
        "Tommie wakes up to the sound of a noisy construction site outside his window.",
        "Tommie gets out of bed and heads to the kitchen to make himself some coffee.",
        "Tommie realizes he forgot to buy coffee filters and starts rummaging through his moving boxes to find some.",
        "Tommie finally finds the filters and makes himself a cup of coffee.",
        "The coffee tastes bitter, and Tommie regrets not buying a better brand.",
        "Tommie checks his email and sees that he has no job offers yet.",
        "Tommie spends some time updating his resume and cover letter.",
        "Tommie heads out to explore the city and look for job openings.",
        "Tommie sees a sign for a job fair and decides to attend.",
        "The line to get in is long, and Tommie has to wait for an hour.",
        "Tommie meets several potential employers at the job fair but doesn't receive any offers.",
        "Tommie leaves the job fair feeling disappointed.",
        "Tommie stops by a local diner to grab some lunch.",
        "The service is slow, and Tommie has to wait for 30 minutes to get his food.",
        "Tommie overhears a conversation at the next table about a job opening.",
        "Tommie asks the diners about the job opening and gets some information about the company.",
        "Tommie decides to apply for the job and sends his resume and cover letter.",
        "Tommie continues his search for job openings and drops off his resume at several local businesses.",
        "Tommie takes a break from his job search to go for a walk in a nearby park.",
        "A dog approaches and licks Tommie's feet, and he pets it for a few minutes.",
        "Tommie sees a group of people playing frisbee and decides to join in.",
        "Tommie has fun playing frisbee but gets hit in the face with the frisbee and hurts his nose.",
        "Tommie goes back to his apartment to rest for a bit.",
        "A raccoon tore open the trash bag outside his apartment, and the garbage is all over the floor.",
        "Tommie starts to feel frustrated with his job search.",
        "Tommie calls his best friend to vent about his struggles.",
        "Tommie's friend offers some words of encouragement and tells him to keep trying.",
        "Tommie feels slightly better after talking to his friend.",
    ]

    # Let's send Tommie on their way. We'll check in on their summary every few observations to watch it evolve.
    for i, observation in enumerate(observations):
        _, reaction = tommie.generate_reaction(observation)
        print(colored(observation, "green"), reaction)
        if ((i + 1) % 20) == 0:
            print('*' * 40)
            print(colored(f"After {i+1} observations, Tommie's summary is:\n{tommie.get_summary(force_refresh=True)}", "blue"))
            print('*' * 40)

    print(interview_agent(tommie, "Tell me about how your day has been going"))
    print(interview_agent(tommie, "How do you feel about coffee?"))
    print(interview_agent(tommie, "Tell me about your childhood dog!"))

    eve = GenerativeAgent(name="Eve",
                          age=34,
                          traits="curious, helpful",  # You can add more persistent traits here.
                          status="N/A",  # When connected to a virtual world, we can have the characters update their status.
                          memory_retriever=create_new_memory_retriever(),
                          llm=LLM,
                          daily_summaries=["Eve started her new job as a career counselor last week and received her first assignment, a client named Tommie."],
                          reflection_threshold=5)

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%A %B %d")
    eve_memories = [
        "Eve overhears her colleague say something about a new client being hard to work with",
        "Eve wakes up and hear's the alarm",
        "Eve eats a boal of porridge",
        "Eve helps a coworker on a task",
        "Eve plays tennis with her friend Xu before going to work",
        "Eve overhears her colleague say something about Tommie being hard to work with",
        
    ]
    for memory in eve_memories:
        eve.add_memory(memory)
    print(eve.get_summary())

    print(interview_agent(eve, "How are you feeling about today?"))
    print(interview_agent(eve, "What do you know about Tommie?"))
    print(interview_agent(eve, "Tommie is looking to find a job. What are are some things you'd like to ask him?"))
    print(interview_agent(eve, "You'll have to ask him. He may be a bit anxious, so I'd appreciate it if you keep the conversation going and ask as many questions as possible."))

    # Run a simple conversation between Tommie and Eve.
    agents = [tommie, eve]
    run_conversation(agents, "Tommie said: Hi, Eve. Thanks for agreeing to share your story with me and give me advice. I have a bunch of questions.")

    # We can see a current "Summary" of a character based on their own perception of self
    # has changed
    print(tommie.get_summary(force_refresh=True))
    print(eve.get_summary(force_refresh=True))
    print(interview_agent(tommie, "How was your conversation with Eve?"))
    print(interview_agent(eve, "How was your conversation with Tommie?"))
    print(interview_agent(eve, "What do you wish you would have said to Tommie?"))
    print(interview_agent(tommie, "What happened with your coffee this morning?"))
