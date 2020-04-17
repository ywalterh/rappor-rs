mod client;

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
pub async fn greet(input: String)  -> Result<JsValue, JsValue> {
    init_panic_hook();
    // add more code here
    let f = client::Factory::new(0.01);
    let result = f.process(input);
    // use rest API call to communicate with server
    let res = reqwest::Client::new()
        .post("http://localhost:8000/api/report")
        .body(result)
        .header("Access-Control-Allow-Origin", "*")
        .send()
        .await?;
    
    let text = res.text().await?;
    alert(format!("Reported {}", text).as_str());
    Ok(JsValue::from_str(&text))
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
