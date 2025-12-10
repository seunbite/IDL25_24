from typing import List, Dict, Any, Tuple
import graphviz
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map
import random
import os

class LLMAgent:
    def __init__(self, model_name: str, max_memory: dict | None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if max_memory:
            infer_auto_device_map(self.model, max_memory=max_memory)
            
    def generate(self, input_text: str, system_message: str | None, max_length: int = 512):
        if system_message:
            messages = self.tokenizer.apply_chat_template([{'role' : 'system', 'content' : system_message}, {'role' : 'user', 'content' : input_text}], tokenize=False, add_generation_prompt=True)
        else:
            messages = self.tokenizer.apply_chat_template([{'role' : 'user', 'content' : input_text}], tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(messages, return_tensors='pt').to(self.model.device)
        output = self.model.generate(**input_ids, max_new_tokens=max_length, num_return_sequences=1)
        output_text = self.tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        print(f'-----------\n{messages}\n----\n{output_text}')
        return output_text
    
    
class CareerState:
    def __init__(self,
                 current_position: str,
                 skills: List[str],
                 years_experience: int,
                 education_level: str,
                 node_id: int = 0,
                 graph_id: str = "",
                 parent: 'CareerState' = None):  # parent 추가
        self.position = current_position
        self.skills = skills
        self.years_experience = years_experience
        self.education_level = education_level
        self.node_id = node_id
        self.graph_id = graph_id
        self.parent = parent  # parent 추가
        
    def __str__(self):
        return f"{self.position} (Exp:{self.years_experience}y, Edu:{self.education_level})"


class CareerNode:
    def __init__(self, state: CareerState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.action_taken = None
        self.values = {}  # 다양한 가치 지표 저장
        
    def traverse(self):
        yield self  # 현재 노드 반환
        for child in self.children:  # 모든 자식 노드에 대해
            yield from child.traverse()  # 재귀적으로 순회


class MCTSVisualizer:
    def __init__(self):
        """Initialize visualizer with graphviz configuration"""
        self.reset_graph()
    
    def reset_graph(self):
        """Reset the graph for new visualization"""
        self.dot = graphviz.Digraph(comment='Career MCTS Tree')
        self.dot.attr(rankdir='TB')  # Top to Bottom direction
        self.node_count = 0

    def create_node_label(self, idx: int, node: Dict) -> str:
        """Create formatted label for a node
        
        Args:
            idx: Node index
            node: Node data containing position, experience, and values
        """
        # Truncate position if too long
        position = node['position']
        if len(position) > 30:
            position = position[:27] + "..."
            
        return f"""#{idx}
{position}
Exp: {node['years_experience']}yr
Salary: ${node['values']['expected_salary']/1000:.1f}k
Fit: {node['values']['career_fit']}%"""

    def visualize_career_tree(self, 
                            data: List[Dict], 
                            save_path: List[str],
                            ) -> None:
        """Visualize career trees
        
        Args:
            data: List of career tree data
            save_path: Path to save the visualization
            sample_size: Number of trees to visualize (randomly sampled if specified)
        """

        # Process each tree
        for graph in data:
            self.reset_graph()  # Reset for each new tree
            
            # Add initial node
            self.dot.node('0', 
                         self.create_node_label(0, graph['nodes'][0]),
                         style='filled', 
                         fillcolor='lightblue')
            
            # Add other nodes and edges
            for idx, node in enumerate(graph['nodes']):
                if idx == 0:  # Skip root node
                    continue
                    
                # Create node with color based on experience
                node_color = f"0.{min(node['years_experience'], 9)} 0.3 1.0"
                self.dot.node(str(idx),
                            self.create_node_label(idx, node),
                            style='filled',
                            fillcolor=node_color)
                
                # Add edges to children
                for child_idx in node['children_idx']:
                    self.dot.edge(str(idx), str(child_idx))
            
            # Save the visualization
            tree_save_path = save_path+f"_{graph['graph_id']}"
            self.dot.render(tree_save_path, view=False, format='png')
            print(f"Saved to {tree_save_path}.png")

