"""LangGraph node factories for CloudGuard integration."""

from langchain_core.runnables import RunnableLambda
from ...runtime.input_gate import InputCloudGate
from ...runtime.output_gate import OutputCloudGate
from .state_keys import *

def make_input_gate_node(gate: InputCloudGate, text_key: str = USER_INPUT):
    """
    Create a LangGraph node that applies input routing using CloudGuard.
    
    Args:
        gate: Configured InputCloudGate instance
        text_key: State key containing the input text to route
        
    Returns:
        RunnableLambda: Node that adds routing decision to state
    """
    def _route_input(state):
        text = state.get(text_key, "")
        result = gate.route(text)
        return {CLOUDGUARD_INPUT: result.__dict__}
    
    return RunnableLambda(_route_input)

def make_output_gate_node(gate: OutputCloudGate,
                          input_key: str = USER_INPUT,
                          output_key: str = LLM_OUTPUT):
    """
    Create a LangGraph node that validates LLM output using CloudGuard.
    
    Args:
        gate: Configured OutputCloudGate instance
        input_key: State key containing original user input
        output_key: State key containing LLM response to validate
        
    Returns:
        RunnableLambda: Node that adds validation result to state
    """
    def _validate_output(state):
        user_text = state.get(input_key, "")
        llm_text = state.get(output_key, "")
        result = gate.validate(user_text, llm_text)
        
        return {CLOUDGUARD_OUTPUT: {
            "ok": result.ok,
            "kept_text": result.kept_text,
            "coverage": result.coverage,
            "dropped_segments": result.dropped_segments,
            "coverage_ratio": result.coverage_ratio,
            "scores_summary": result.scores_summary
        }}
    
    return RunnableLambda(_validate_output)