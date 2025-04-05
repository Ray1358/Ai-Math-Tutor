package com.baezcostiganreed.mathtutorapp;

import org.springframework.ai.chat.memory.InMemoryChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;


/**
 * Provides REST endpoints to interact with a chat client.
 */
@RestController
public class ChatController {
    private static final int TOP_K = 5;
    private static final double SIMILARITY_THRESHOLD = .6;
    private SystemPromptTemplate systemPromptTemplate;
    private final OllamaApi ollamaApi = new OllamaApi("http://localhost:11434");
    private final PgVectorStore vectorStore;
    String conversationId = UUID.randomUUID().toString();
    InMemoryChatMemory chatMemory = new InMemoryChatMemory();


    @Value("classpath:/prompts/prompt_template.txt")
    private Resource systemTemplateResource;



    /**
     * Constructs a new controller with a vector store.
     *
     * @param vectorStore The vector store used for vector operations
     */
    public ChatController(PgVectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    /**
     * Handles GET requests with an optional user message.
     * Sends the user message to the chat client and returns the chat response.
     *
     * @param topic  the desired math topic of the chat
     * @param usermessage  the text provided by the user
     * @return the chat client's response or an error message
     */
    @GetMapping("/chat")
    public Flux<OllamaApi.ChatResponse> chat(@RequestParam(value = "topic", defaultValue = "linear equations") String topic,
                                             @RequestParam(value = "usermessage", defaultValue = " ") String usermessage) {
        String filterExpressionChapter = "";
        String chapterContent = "";
        switch (topic) {
            case "linear equations":
                filterExpressionChapter = "file_name == 'fundamentals-of-mathematics.pdf' && page_number >= 1249 && page_number <= 1357 || file_name == 'Beginning_and_Intermediate_Algebra.pdf' && page_number >= 28 && page_number <= 79";
                break;
            case "fractions":
                filterExpressionChapter = "file_name == 'fundamentals-of-mathematics.pdf' && page_number >= 400 && page_number <= 645 || file_name == 'Beginning_and_Intermediate_Algebra.pdf' && page_number >= 12 && page_number <= 17";
                break;
            case "integers":
                filterExpressionChapter = "file_name == 'fundamentals-of-mathematics.pdf' && page_number >= 15 && page_number <= 268 || file_name == 'Beginning_and_Intermediate_Algebra.pdf' && page_number >= 7 && page_number <= 9";
                break;
            case "real numbers":
                filterExpressionChapter = "file_name == 'fundamentals-of-mathematics.pdf' && page_number >= 1131 && page_number <= 1144";
                break;
            case "signed numbers":
                filterExpressionChapter = "file_name == 'fundamentals-of-mathematics.pdf' && page_number >= 1145 && page_number <= 1220";
                break;
            case "decimals":
                filterExpressionChapter = "file_name == 'fundamentals-of-mathematics.pdf' && page_number >= 663 && page_number <= 828";
                break;
            case "percents":
                filterExpressionChapter = "file_name == 'fundamentals-of-mathematics.pdf' && page_number >= 848 && page_number <= 920";
                break;
            case "polynomials":
                filterExpressionChapter = "file_name == 'Beginning_and_Intermediate_Algebra.pdf' && page_number >= 177 && page_number <= 205";
                break;
            case "factoring":
                filterExpressionChapter = "file_name == 'Beginning_and_Intermediate_Algebra.pdf' && page_number >= 212 && page_number <= 237";
                break;
        }
        try {
            String systemPrompt = String.format(systemTemplateResource.getContentAsString(Charset.defaultCharset()), topic);
            Message userMessage = new UserMessage(usermessage);
            chatMemory.add(conversationId, userMessage);

            List<Message> messages = chatMemory.get(conversationId, 50);
            String chatHistory = messages.stream()
                    .map(Message::getText)
                    .collect(Collectors.joining(" "));

            List<Document> chapterResults = vectorStore.similaritySearch(
                    SearchRequest.builder()
                            .query("Steps to find the solution. How to Solve. " + topic + " " + chatHistory)
                            .topK(TOP_K)
                            .similarityThreshold(SIMILARITY_THRESHOLD)
                            .filterExpression(filterExpressionChapter)
                            .build());

            if (chapterResults != null && !chapterResults.isEmpty()) {
                chapterContent = chapterResults.stream().map(Document::getText).collect(Collectors.joining("\n "));
            } else {
                chapterContent = "No documents provided \n";
            }

            List<OllamaApi.Message> ollamaMessages = messages.stream().map(message -> {
                OllamaApi.Message.Role role;
                if (message instanceof UserMessage) {
                    role = OllamaApi.Message.Role.USER;
                } else if (message instanceof AssistantMessage) {
                    role = OllamaApi.Message.Role.ASSISTANT;
                } else {
                    role = OllamaApi.Message.Role.SYSTEM;
                }
                return OllamaApi.Message.builder(role)
                        .content(message.getText())
                        .build();
            }).collect(Collectors.toList());
            ollamaMessages.add(0, OllamaApi.Message.builder(OllamaApi.Message.Role.SYSTEM)
                    .content(systemPrompt)
                    .build());
            ollamaMessages.add(1,OllamaApi.Message.builder(OllamaApi.Message.Role.TOOL)
                    .content(chapterContent)
                    .build());

            OllamaApi.ChatRequest request = OllamaApi.ChatRequest.builder("phi4-mini")
                    .stream(true)
                    .messages(ollamaMessages)
                    .options(OllamaOptions.builder()
                            .numCtx(8000)
                            .temperature(.25)
                            .topP(.6)
                            .build())
                    .build();

            return this.ollamaApi.streamingChat(request)
                    .doOnNext(ollamaChatResponse -> {
                        String responseContent = ollamaChatResponse.message().content();
                        Message assistantMessage = new AssistantMessage(responseContent);
                        chatMemory.add(conversationId, assistantMessage);
                    });

        } catch (IOException e) {
            return Flux.error(new RuntimeException("Error reading system prompt template: " + e.getMessage(), e));
        } catch (Exception e) {
            return Flux.error(new RuntimeException("Error processing chat request: " + e.getMessage(), e));
        }
    }}