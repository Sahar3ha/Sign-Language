import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import VideoRecognition from './Pages/VideoRecon';
import ApiTest from './Pages/Apitest';
import VoiceRecognition from './Pages/VoiceRecon';

function App() {
  return (
    <Router>
      <Routes>
        <Route path='/videorecon' element={<VideoRecognition />} />
        <Route path='/voicerecon' element={<VoiceRecognition />} />

        <Route path="/hello" element={<ApiTest/>} />
      </Routes>
    </Router>
  );
}

export default App;
