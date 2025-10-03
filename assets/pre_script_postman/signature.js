// Siapkan variable signature_secret pada collection Variaables
const secretKey = pm.collectionVariables.get("signature_secret");

if (secretKey) {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    
    let canonicalJson = "";
    
    if (pm.request.body && pm.request.body.raw) {
        let rawBody = pm.request.body.raw;
        
        // New: Regex to remove single-line (//) and multi-line (/* */) comments
        const cleanedJsonString = rawBody.replace(/\/\/.*|\/\*[\s\S]*?\*\//g, '');
        
        try {
            // Parse the cleaned string instead of the raw body
            const bodyJson = JSON.parse(cleanedJsonString);
            canonicalJson = JSON.stringify(bodyJson);
        } catch (e) {
            console.error("The body is still not valid JSON after removing comments:", e);
            console.error("Cleaned JSON string that failed to parse:", cleanedJsonString);
        }
    }

    const messageToSign = `${timestamp}.${canonicalJson}`;
    const hash = CryptoJS.HmacSHA256(messageToSign, secretKey);
    const signatureHex = hash.toString(CryptoJS.enc.Hex);
    const origin = pm.collectionVariables.get("origin");

    pm.request.headers.add({ key: 'x-signature', value: signatureHex });
    pm.request.headers.add({ key: 'x-timestamp', value: timestamp});
    pm.request.headers.add({ key: 'origin', value: origin});
}
