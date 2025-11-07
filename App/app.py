from transformers import pipeline

# Load a free Hugging Face model for text generation
# (distilgpt2 is small and fast; later you can upgrade to bigger models)
feedback_model = pipeline("text-generation", model="distilgpt2")

def get_resume_feedback(resume_json):
    """
    Generate simple resume feedback from extracted JSON resume data.
    
    Args:
        resume_json (dict): Parsed resume info from ResumeParser.
    
    Returns:
        str: AI-generated feedback text.
    """
    # Convert structured JSON into a readable text summary
    resume_text = "\n".join([f"{k}: {v}" for k, v in resume_json.items() if v])

    # Build prompt for the model
    prompt = f"""
    You are a resume expert. Analyze the following resume details and suggest 3 improvements.
    Focus on clarity, action verbs, ATS friendliness, and missing skills.
    
    Resume:
    {resume_text}
    """

    try:
        response = feedback_model(prompt, max_length=250, num_return_sequences=1, do_sample=True)
        feedback = response[0]["generated_text"]
        return feedback
    except Exception as e:
        return f"Error generating feedback: {e}"
