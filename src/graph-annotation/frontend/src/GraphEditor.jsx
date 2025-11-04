import Dagre from '@dagrejs/dagre';
import { useState, useEffect, useCallback, useRef } from "react";
import {
    ReactFlow,
    ReactFlowProvider,
    Panel,
    Background,
    Controls,
    MiniMap,
    addEdge,
    reconnectEdge,
    Edge,
    Node,
    MarkerType,
    useNodesState,
    useEdgesState,
    useReactFlow,
} from '@xyflow/react';
import { fetchData, sendData } from './api'
import NodeWithScrew from './Node'

import '@xyflow/react/dist/style.css';

const nodeTypes = { nodeWithScrew: NodeWithScrew };


// copied from https://reactflow.dev/learn/layouting/layouting
// customized to rmove edge: https://reactflow.dev/examples/edges/delete-edge-on-drop
const getLayoutedElements = (nodes, edges, options) => {
    const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(
        () => ({})
    );
    g.setGraph({ rankdir: options.direction });

    // conncted node IDs
    const connectedNodeIds = new Set(
        edges.flatMap(edge => [edge.source, edge.target])
    );

    // Separate connected and isolated nodes
    const connectedNodes = nodes.filter(
        node => connectedNodeIds.has(node.id)
    );
    const isolatedNodes = nodes.filter(
        node => !connectedNodeIds.has(node.id)
    );

    edges.forEach((edge) => g.setEdge(edge.source, edge.target));
    connectedNodes.forEach((node) => g.setNode(node.id, {
        ...node,
        width: node.measured?.width ?? 0,
        height: node.measured?.height ?? 0,
    }));

    Dagre.layout(g);

    // Apply layout to connected nodes, keep isolated nodes unchanged
    const layoutedNodes = connectedNodes.map(node => {
        const position = g.node(node.id);
        const x = position.x - (node.measured?.width ?? 0) / 2;
        const y = position.y - (node.measured?.height ?? 0) / 2;
        return { ...node, position: { x, y } };
    });

    return {
        // Merge back isolated nodes with unchanged positions
        nodes: [...layoutedNodes, ...isolatedNodes],
        edges,
    };
};

// make sure to use different variable names even across files
// export default function DAGWrapper ({ propNodes, propEdges }) {
export default function DAGWrapper ({
    idx, nodes, edges, setNodes, setEdges, onNodesChange, onEdgesChange
}) {

    const { fitView } = useReactFlow();
    const edgeReconnectSuccessful = useRef(true);
    const prevIdxRef = useRef(0);

    const onConnect = useCallback((params) => {
        setEdges((eds) => addEdge({
            ...params,
            markerEnd: { type: 'arrow' }
        }, eds));
    }, [setEdges]);

    const onReconnectStart = useCallback(() => {
        edgeReconnectSuccessful.current = false;
    }, []);

    const onReconnect = useCallback((oldEdge, newConnection) => {
        edgeReconnectSuccessful.current = true;
        setEdges((els) => reconnectEdge(oldEdge, newConnection, els));
    }, []);

    const onReconnectEnd = useCallback((_, edge) => {
        if (!edgeReconnectSuccessful.current) {
            setEdges((eds) => eds.filter((e) => e.id !== edge.id));
        }
        edgeReconnectSuccessful.current = true;
    }, []);

    const onLayout = useCallback(
        (direction) => {
            const layouted = getLayoutedElements(nodes, edges, { direction });
            setNodes([...layouted.nodes]);
            setEdges([...layouted.edges]);
            window.requestAnimationFrame(() => {
                fitView();
            });
        },
        [nodes, edges],
    );


    return (
        <div>
            <div style={{ width: "100%", height: "85vh", border: "1px solid black" }}>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    snapToGrid
                    onReconnect={onReconnect}
                    onReconnectStart={onReconnectStart}
                    onReconnectEnd={onReconnectEnd}
                    onConnect={onConnect}
                    nodeTypes={nodeTypes}
                    fitView
                >
                    <Panel position="bottom-left" style={{ left: '40px' }}>
                        <button className="btn btn-outline-info" onClick={() => onLayout('TB')}>Layout</button>
                    </Panel>
                    <MiniMap />
                    <Controls />
                    <Background />
                </ReactFlow>
            </div>
        </div>
    );
};
