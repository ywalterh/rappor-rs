use client::encode;
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub async fn greet(input: String) -> Result<JsValue, JsValue> {
    init_panic_hook();
    // add more code here
    let f = encode::EncoderFactory::new(1);
    let result = f.encode(1, input);
    // use rest API call to communicate with server
    let res = reqwest::Client::new()
        .post("http://localhost:8000/api/report")
        .body(result)
        .header("Access-Control-Allow-Origin", "*")
        .send()
        .await;
    match res {
        Ok(res) => {
            let text = res.text().await;
            match text {
                Ok(text) => {
                    alert(format!("Reported {}", text).as_str());
                    Ok(JsValue::from_str(&text))
                }
                Err(err) => Err(JsValue::from_str(err.to_string().as_str())),
            }
        }
        Err(err) => Err(JsValue::from_str(err.to_string().as_str())),
    }
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
