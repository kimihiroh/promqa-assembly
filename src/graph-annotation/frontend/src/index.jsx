// this file is for rendering by convention
import React from "react";
import ReactDOM from "react-dom/client";  // Import createRoot from React 18
import {ReactFlowProvider} from '@xyflow/react';
import App from "./App";

import '@xyflow/react/dist/style.css';

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
    <React.StrictMode>
        <ReactFlowProvider>
            <App />
        </ReactFlowProvider>
    </React.StrictMode>
);
