const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

/**
 * Chatterbox API ile ses dosyası oluşturma örneği
 */
async function generateSpeech(text, outputPath = 'output.wav') {
    const url = 'http://YOUR_SERVER_IP:8000/generate'; // Dokploy IP'nizi buraya yazın

    const form = new FormData();
    form.append('text', text);
    form.append('temperature', '0.8');
    // Eğer kendi sesinizi klonlamak isterseniz:
    // form.append('audio_prompt', fs.createReadStream('referans_ses.wav'));

    try {
        console.log('Ses oluşturuluyor...');
        const response = await axios.post(url, form, {
            headers: {
                ...form.getHeaders(),
            },
            responseType: 'stream'
        });

        const writer = fs.createWriteStream(outputPath);
        response.data.pipe(writer);

        return new Promise((resolve, reject) => {
            writer.on('finish', () => {
                console.log(`Ses dosyası başarıyla kaydedildi: ${outputPath}`);
                resolve(outputPath);
            });
            writer.on('error', reject);
        });
    } catch (error) {
        console.error('Hata oluştu:', error.message);
        if (error.response) {
            console.error('API Yanıtı:', error.response.data);
        }
    }
}

// Kullanım:
// generateSpeech('Selam dostum, bu ses Chatterbox tarafından oluşturuldu!');
