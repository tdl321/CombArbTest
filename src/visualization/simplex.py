"""Simplex geometry for probability visualization."""
import numpy as np
from typing import Optional

class SimplexProjector:
    """Project N-dimensional probability vectors to 2D for visualization."""
    
    def __init__(self, n_markets: int, market_labels: Optional[list[str]] = None):
        self.n = n_markets
        self.labels = market_labels or [f"M{i}" for i in range(n_markets)]
        self.vertices = self._compute_vertices()
    
    def _compute_vertices(self) -> np.ndarray:
        """Compute 2D vertices for simplex visualization."""
        if self.n == 2:
            # Line segment
            return np.array([[0.0, 0.0], [1.0, 0.0]])
        elif self.n == 3:
            # Equilateral triangle
            return np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2]
            ])
        elif self.n == 4:
            # Square projection of tetrahedron
            return np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0]
            ])
        else:
            # Regular polygon for N>4
            angles = np.linspace(0, 2*np.pi, self.n, endpoint=False)
            return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def to_2d(self, probs: np.ndarray) -> np.ndarray:
        """Convert probability vector to 2D point via barycentric coords."""
        probs = np.asarray(probs)
        if probs.sum() > 0:
            # Normalize to handle points outside simplex
            normalized = probs / probs.sum()
        else:
            normalized = np.ones(self.n) / self.n
        return normalized @ self.vertices
    
    def to_2d_unnormalized(self, probs: np.ndarray) -> np.ndarray:
        """Convert without normalizing - shows points outside simplex."""
        probs = np.asarray(probs)
        # Scale vertices by probability values
        return probs @ self.vertices
    
    def is_feasible(self, probs: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if probability vector is inside simplex."""
        probs = np.asarray(probs)
        return np.all(probs >= -tol) and np.abs(probs.sum() - 1.0) < tol
    
    def distance_to_simplex(self, probs: np.ndarray) -> float:
        """Compute L2 distance from point to simplex."""
        probs = np.asarray(probs)
        # Project to simplex (normalize)
        projected = np.maximum(probs, 0)
        if projected.sum() > 0:
            projected = projected / projected.sum()
        else:
            projected = np.ones(self.n) / self.n
        return np.linalg.norm(probs - projected)
    
    def get_simplex_boundary(self, n_points: int = 100) -> np.ndarray:
        """Get points along simplex boundary for plotting."""
        if self.n == 2:
            return self.vertices
        elif self.n == 3:
            # Triangle edges
            points = []
            for i in range(3):
                v1, v2 = self.vertices[i], self.vertices[(i+1) % 3]
                for t in np.linspace(0, 1, n_points // 3):
                    points.append(v1 * (1-t) + v2 * t)
            return np.array(points)
        else:
            # Polygon edges
            points = []
            for i in range(self.n):
                v1, v2 = self.vertices[i], self.vertices[(i+1) % self.n]
                for t in np.linspace(0, 1, n_points // self.n):
                    points.append(v1 * (1-t) + v2 * t)
            return np.array(points)
