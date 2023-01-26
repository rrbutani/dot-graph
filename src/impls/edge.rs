use crate::structs::Edge;

impl Edge {
    pub fn to_dot(&self, indent: usize) -> String {
        let tabs = "\t".repeat(indent);
        let mut dot = String::from("");

        let headport = match self.attrs.get("headport") {
            Some(headport) => format!(":{}", headport),
            None => "".to_string(),
        };

        let tailport = match self.attrs.get("tailport") {
            Some(tailport) => format!(":{}", tailport),
            None => "".to_string(),
        };

        dot.push_str(&format!("{}{}{} -> {}{}", tabs, self.from, tailport, self.to, headport));

        let mut attrs = self.attrs.clone();
        attrs.remove("headport");
        attrs.remove("tailport");

        if attrs.len() > 0 {
            dot.push_str(" [ ");
            for (key, value) in &attrs {
                dot.push_str(&format!("{}={} ", key, value));
            }
            dot.push_str("]");
        }

        dot
    }
}
