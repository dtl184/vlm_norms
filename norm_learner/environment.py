from __future__ import annotations

from abc import ABC, abstractmethod

from .types import Action, State, Trajectory, WorldState


class EnvironmentInterface(ABC):
    """
    Contract that every environment adapter must satisfy.

    The symbolic algorithm only calls ``plan``, ``shortest_path_cost``, and
    ``trajectory_cost``.  The grounding layer additionally calls
    ``describe_state``, ``describe_action``, ``extract_semantic_features``,
    and ``get_environment_description``.

    To plug in a new environment (MazeNamo, ROSNav, etc.) subclass this,
    implement the abstract methods, and pass an instance to ``NormLearner``.
    """

    # ------------------------------------------------------------------
    # Planning interface — required by the symbolic algorithm
    # ------------------------------------------------------------------

    @abstractmethod
    def plan(self, start: State, goal: State) -> list[Trajectory]:
        """Return one or more cost-optimal trajectories from start to goal."""

    @abstractmethod
    def shortest_path_cost(self, start: State, goal: State) -> float:
        """Return the cost of the shortest unconstrained path."""

    @abstractmethod
    def trajectory_cost(self, trajectory: Trajectory) -> float:
        """Return the cost of the given trajectory."""

    def segment_cost(self, trajectory: Trajectory, i: int, j: int) -> float:
        return self.trajectory_cost(trajectory[i:j])

    def successor(self, state: State, action: Action) -> State | None:
        """Return the state reached by taking *action* in *state*, or None."""
        return None  # subclasses override when they have a transition model

    def describe_transition(self, pre: State, action: Action, post: State) -> str | None:
        """
        Return a human-readable semantic label for a (pre, action, post) triple,
        or None to fall back to the generic action label.

        Override in environment subclasses to give meaningful names to transitions
        whose semantics differ depending on context (e.g. INTERACT can mean
        "pick up basket", "pick up item", or "pay at checkout").
        """
        return None

    # ------------------------------------------------------------------
    # Semantic interface — required by the grounding layer
    # ------------------------------------------------------------------

    @abstractmethod
    def describe_state(self, state: State) -> str:
        """Short human-readable label for a state (used in VLM prompt)."""

    @abstractmethod
    def describe_action(self, action: Action) -> str:
        """Short human-readable label for an action."""

    @abstractmethod
    def extract_semantic_features(self, state: State) -> WorldState:
        """
        Return a flat dict of semantic features for *state*.

        Examples
        --------
        Grid world  : {"position": (x, y), "zone": "aisle_3", "near_shelf": True}
        Social nav  : {"position": (px, py), "velocity": (vx, vy), "n_nearby_agents": 2}
        """

    @abstractmethod
    def get_environment_description(self) -> str:
        """
        A paragraph describing the environment that is prepended to every VLM
        prompt so the model has context about what it is observing.
        """
