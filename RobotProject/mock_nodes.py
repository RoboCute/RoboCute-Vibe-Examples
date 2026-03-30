# -*- coding: utf-8 -*-
"""
Mock RBC Nodes Module

当没有安装 robocute 时使用的兼容模块
"""

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NodeInput:
    """节点输入定义"""
    name: str
    type: str
    required: bool = True
    default: Optional[Any] = None
    description: str = ""


@dataclass
class NodeOutput:
    """节点输出定义"""
    name: str
    type: str
    description: str = ""


class RBCNode(ABC):
    """节点基类（Mock）"""
    
    NODE_TYPE: str = "base_node"
    DISPLAY_NAME: str = "Base Node"
    CATEGORY: str = "default"
    DESCRIPTION: str = ""
    
    def __init__(self, node_id: str, context=None):
        self.node_id = node_id
        self.context = context
        self._inputs: Dict[str, Any] = {}
        self._outputs: Dict[str, Any] = {}
    
    @classmethod
    @abstractmethod
    def get_inputs(cls) -> List[NodeInput]:
        pass
    
    @classmethod
    @abstractmethod
    def get_outputs(cls) -> List[NodeOutput]:
        pass
    
    def set_input(self, name: str, value: Any) -> None:
        self._inputs[name] = value
    
    def get_input(self, name: str, default: Any = None) -> Any:
        return self._inputs.get(name, default)
    
    def set_output(self, name: str, value: Any) -> None:
        self._outputs[name] = value
    
    def get_output(self, name: str) -> Any:
        return self._outputs.get(name)
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        pass
    
    def run(self) -> Dict[str, Any]:
        outputs = self.execute()
        for name, value in outputs.items():
            self.set_output(name, value)
        return outputs


def register_node(cls):
    """节点注册装饰器（Mock）"""
    return cls
