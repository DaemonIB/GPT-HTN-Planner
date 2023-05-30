import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

function HTNPlanner() {
  const [taskNode, setTaskNode] = useState(null);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('task_node_update', (data) => {
      setTaskNode(data);
    });

    return () => newSocket.close();
  }, []);

  // Render the task node as a tree
  const renderTaskNode = (node) => {
    if (!node) return null;
    return (
      <li>
        {node.task_name} ({node.status}) {/* Display the task status */}
        {node.children.length > 0 && (
          <ul>
            {node.children.map((child) => renderTaskNode(child))}
          </ul>
        )}
      </li>
    );
  };

  return (
    <div>
      <h1>HTN Planner Visualization</h1>
      <ul>{renderTaskNode(taskNode)}</ul>
    </div>
  );
}

export default HTNPlanner;