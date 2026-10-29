# \# LegalLens AI

# \## Legal Document Analysis and Question Answering Using RAG and Agentic AI

# 

# LegalLens AI is a full-stack legal-document analysis application that helps users explore PDF contracts, retrieve relevant clauses, ask natural-language questions, generate document summaries, review potential contractual risks, and classify individual clauses.

# 

# The project combines \*\*Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), LangChain, LangGraph, vector search, and machine-learning-based clause classification\*\* in a React and FastAPI application.

# 

# > \*\*Disclaimer:\*\* LegalLens AI is an educational project. Its outputs may be incomplete or incorrect and must not be treated as legal advice.

# 

# \---

# 

# \## 1. Problem Statement

# 

# Legal agreements can be lengthy and difficult to review manually. Locating specific terms, understanding contractual obligations, and identifying unclear or missing provisions can require considerable time.

# 

# LegalLens AI aims to make document exploration easier through evidence-grounded question answering, automated summaries, potential-risk screening, and machine-learning-based clause classification.

# 

# \## 2. Features

# 

# \- \*\*PDF upload and indexing:\*\* Extract text and page information from PDF documents and prepare it for retrieval.

# \- \*\*Document question answering:\*\* Ask natural-language questions about an uploaded contract.

# \- \*\*Retrieval-Augmented Generation:\*\* Retrieve relevant document passages and supply them to the language model as context.

# \- \*\*Page citations:\*\* Return page references where supported by retrieved document evidence.

# \- \*\*Document summarization:\*\* Generate structured summaries of key terms, obligations, dates, and unclear provisions.

# \- \*\*Potential-risk review:\*\* Highlight contractual clauses and omissions that may deserve further examination.

# \- \*\*LangGraph agent workflow:\*\* Route user requests to question-answering, summarization, or risk-review workflows.

# \- \*\*LangChain integration:\*\* Use LangChain's ChatGroq integration for language-model generation.

# \- \*\*Clause classification:\*\* Predict a general contractual clause category using a trained TF-IDF and LinearSVC model.

# \- \*\*Evaluation metrics:\*\* Store validation and test results for the trained clause classifier.

# \- \*\*Interactive web interface:\*\* Use a React and TypeScript frontend to upload documents, ask questions, view reports, classify clauses, and interact with the agent.

# 

# \## 3. Technology Stack

# 

# | Component | Technology |

# |---|---|

# | Frontend | React, TypeScript, Vite |

# | Styling | CSS |

# | Backend API | Python, FastAPI |

# | PDF text extraction | PyPDF |

# | Text embeddings | Sentence Transformers |

# | Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |

# | Vector database | Qdrant |

# | Metadata storage | SQLite |

# | Language model | Groq-hosted LLM |

# | LLM orchestration | LangChain |

# | Agent workflow orchestration | LangGraph |

# | Groq integration | LangChain-Groq |

# | Clause classification | TF-IDF and LinearSVC |

# | Classification dataset | Filtered single-label subset of LEDGAR |

# | Model persistence | Joblib |

# 

# \## 4. System Architecture

# 

# LegalLens AI contains three primary processing pipelines.

# 

# \### A. Document ingestion and RAG question answering

# 

# 1\. The user uploads a PDF through the React frontend.

# 2\. FastAPI validates the uploaded file and extracts text with page information.

# 3\. The extracted text is divided into smaller chunks.

# 4\. Sentence Transformers generates vector embeddings for the chunks.

# 5\. The chunk vectors and associated text are stored in a local Qdrant collection.

# 6\. Document metadata is stored in SQLite.

# 7\. When the user asks a question, the backend generates a query embedding and retrieves relevant chunks for the selected document.

# 8\. Retrieved passages are supplied to the configured Groq-hosted language model.

# 9\. LangChain's `ChatGroq` integration handles language-model generation.

# 10\. The generated answer and retrieved source passages are returned to the frontend.

# 

# The retrieval and generation process helps ground answers in the uploaded document. However, retrieval can miss relevant evidence, and generated answers must still be checked against the source.

# 

# \### B. LangGraph agent workflow

# 

# The application provides a dedicated agent interface and a FastAPI endpoint, `POST /agent`.

# 

# The current graph uses rule-based keyword matching to select one of three workflow branches:

# 

# 1\. \*\*Question answering:\*\* Uses the existing RAG question-answering service.

# 2\. \*\*Document summarization:\*\* Retrieves relevant passages and generates a structured summary.

# 3\. \*\*Potential-risk review:\*\* Retrieves relevant passages and generates a review of potentially unclear, one-sided, or missing contractual terms.

# 

# The workflow returns the selected action, generated result, and retrieved source passages.

# 

# LangGraph orchestrates the workflow, while LangChain's `ChatGroq` integration handles LLM generation. The current routing is \*\*rule-based rather than autonomous LLM-driven tool selection\*\*. The application demonstrates graph-based workflow orchestration, but further

