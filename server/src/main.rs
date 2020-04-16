#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use] extern crate rocket;

#[get("/")]
fn index() -> &'static str {
    "Hello, world!"
}

// accept binary array as input to feed into decoding and learning
#[post("/api/report", data = "<barray>")]
fn report(barray: String) -> String {
    format!("Accepted data [{}]", barray)
}

#[get("/result")]
fn result() -> &'static str {
    "Here's the result and breakdown"
}

fn main() {
    rocket::ignite().mount("/", routes![index, report, result]).launch();
}