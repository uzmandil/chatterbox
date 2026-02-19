const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

/**
 * Chatterbox API ile ses dosyası oluşturma örneği
 * 
 * @param {string} text - Sentezlenecek metin
 * @param {string} voiceName - Sesin ismi (Örn: 'Achird', 'Aoede')
 * @param {string} gender - Cinsiyet ('Man' veya 'Woman')
 * @param {string} outputPath - Çıktı dosyası yolu
 */
async function generateSpeech(text, voiceName = 'Achird', gender = 'Man', outputPath = 'output.wav') {
    const url = 'http://localhost:8000/generate'; // API URL

    const form = new FormData();
    form.append('text', text);
    form.append('voice_name', voiceName);
    form.append('gender', gender);
    form.append('temperature', '0.8');

    // Not: Tüm sesleri listelemek için GET /voices endpointini kullanabilirsiniz.

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
